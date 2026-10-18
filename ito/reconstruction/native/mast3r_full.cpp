// Python bindings of mast3r_full.cu (see there). Both calls release the GIL.
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> calib_linearize(
    torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs, torch::Tensor K,
    torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj, torch::Tensor valid_match,
    torch::Tensor Q, const int height, const int width, const int pixel_border,
    const double z_eps, const double sigma_pixel, const double sigma_depth,
    const double C_thresh, const double Q_thresh);

std::vector<torch::Tensor> block_solve(
    torch::Tensor As, torch::Tensor ii, torch::Tensor jj, torch::Tensor bs, torch::Tensor kk,
    const int n, const double lm, const double ep);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calib_linearize", &calib_linearize, "upstream calib_proj_kernel, one linearisation",
        pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("block_solve", &block_solve, "upstream SparseBlock solve over 7x7 blocks",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
