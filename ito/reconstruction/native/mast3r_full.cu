// Ito's entry points into MASt3R-SLAM's own Gauss-Newton kernels, for the full (global) backend.
//
// Upstream's gauss_newton_calib() runs the whole solve inside one call that holds the GIL and has
// no room for other factors. Ito's global graph adds the robot's factors (odometry, gravity,
// height, ToF scale) to the same normal equations, so it needs the two halves apart:
//   calib_linearize: one launch of upstream's calib_proj_kernel -> per-edge Hessian and gradient
//   block_solve:     upstream's SparseBlock (Eigen SimplicialLLT) over any list of 7x7 blocks
// Both release the GIL, so tracking keeps running while the global graph solves on its thread.
//
// gn_kernels.cu is upstream's file (CC BY-NC-SA 4.0), as ito.build adapted it (current stream,
// int64 indices); it is compiled here once more, unchanged, so its kernels are in this module.
#include "gn_kernels.cu"

std::vector<torch::Tensor> calib_linearize(
    torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs, torch::Tensor K,
    torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj, torch::Tensor valid_match,
    torch::Tensor Q, const int height, const int width, const int pixel_border,
    const double z_eps, const double sigma_pixel, const double sigma_depth,
    const double C_thresh, const double Q_thresh) {
  TORCH_CHECK(Twc.is_contiguous() && Xs.is_contiguous() && Cs.is_contiguous(), "contiguous poses/points");
  TORCH_CHECK(ii.is_contiguous() && jj.is_contiguous() && idx_ii2jj.is_contiguous(), "contiguous edges");
  TORCH_CHECK(valid_match.is_contiguous() && Q.is_contiguous() && K.is_contiguous(), "contiguous edges");
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  torch::Tensor Hs = torch::zeros({4, num_edges, 7, 7}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, 7}, opts);
  if (num_edges == 0) return {Hs, gs};
  calib_proj_kernel<<<num_edges, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      Twc.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      K.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
      ii.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      jj.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
      height, width, pixel_border, (float)z_eps, (float)sigma_pixel, (float)sigma_depth,
      (float)C_thresh, (float)Q_thresh);
  return {Hs, gs};
}

// Solve (A + ep + lm * diag(A)) x = b for n poses of 7 parameters. Blocks with a negative
// index (pinned poses) are skipped, as upstream's SparseBlock does. Returns x (n, 7) as float64
// on the CPU and whether the factorisation succeeded.
std::vector<torch::Tensor> block_solve(
    torch::Tensor As, torch::Tensor ii, torch::Tensor jj, torch::Tensor bs, torch::Tensor kk,
    const int n, const double lm, const double ep) {
  SparseBlock A(n, 7);
  A.update_lhs(As, ii, jj);
  A.update_rhs(bs, kk);
  Eigen::SparseMatrix<double> L(A.A);
  L.diagonal().array() += ep + lm * L.diagonal().array();
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
  solver.compute(L);
  torch::Tensor x = torch::zeros({n, 7}, torch::TensorOptions().dtype(torch::kFloat64));
  bool ok = solver.info() == Eigen::Success;
  if (ok) {
    Eigen::VectorXd solution = solver.solve(A.b);
    ok = solver.info() == Eigen::Success && solution.allFinite();
    if (ok) std::memcpy(x.data_ptr<double>(), solution.data(), sizeof(double) * n * 7);
  }
  return {x, torch::tensor(ok ? 1 : 0)};
}
