"""Size caching allocators from dedicated free VRAM, retaining compositor headroom."""

from ito.render.gpu import GPU, GPU_HEADROOM_BYTES

TORCH_MEMORY_SHARE = 0.8


def free_memory(cuda_free, gpu):
    if gpu.library is not None:
        gpu.sample()
        return min(cuda_free, gpu.free)
    return cuda_free


def prepare_torch(torch):
    gpu = GPU(torch.cuda.current_device())
    try:
        free, total = torch.cuda.mem_get_info()
        available = max(0, free_memory(free, gpu) - GPU_HEADROOM_BYTES)
        limit = torch.cuda.memory_reserved() + int(available * TORCH_MEMORY_SHARE)
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total))
    finally:
        gpu.close()


def configure_pools(xp, torch):
    gpu = GPU(xp.cuda.runtime.getDevice())
    free, total = xp.cuda.runtime.memGetInfo()
    available = max(0, free_memory(free, gpu) - GPU_HEADROOM_BYTES)
    pool = xp.get_default_memory_pool()
    if torch is not None:
        torch_limit = torch.cuda.memory_reserved() + int(available * TORCH_MEMORY_SHARE)
        torch.cuda.set_per_process_memory_fraction(min(1.0, torch_limit / total))
        cupy_limit = pool.total_bytes() + int(available * (1 - TORCH_MEMORY_SHARE))
    else:
        torch_limit = 0
        cupy_limit = pool.total_bytes() + available
    pool.set_limit(size=max(1, cupy_limit))
    return gpu, torch_limit, cupy_limit
