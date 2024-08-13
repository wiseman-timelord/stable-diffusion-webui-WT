from typing import Any
import torch
import multiprocessing

# Global variable to control the percentage of CPU threads to use
THREAD_PERCENTAGE = 85

# Function to detect the number of CPU threads
def get_cpu_thread_count():
    return multiprocessing.cpu_count()

# Function to set the number of threads based on the percentage of available threads
def set_global_threads(percentage: int):
    global THREAD_PERCENTAGE
    THREAD_PERCENTAGE = percentage
    
    # Detect total available threads
    total_threads = get_cpu_thread_count()
    
    # Calculate threads to use (rounded)
    threads_to_use = max(1, int(total_threads * (THREAD_PERCENTAGE / 100.0)))
    
    # Set PyTorch to use the calculated number of threads
    torch.set_num_threads(threads_to_use)
    torch.set_num_interop_threads(threads_to_use)
    
    # Print the number of threads being used
    print(f"({THREAD_PERCENTAGE}%) of total threads = {threads_to_use} out of {total_threads}")

# Set the default threads to 80% of the available CPU threads
set_global_threads(THREAD_PERCENTAGE)

__all__ = ["autocast"]

class autocast(torch.amp.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is equivalent to ``torch.autocast("cpu", args...)``
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        cache_enabled: bool = True,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cpu"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)
