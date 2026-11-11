"""Synchronized inference-timing and tensor-storage integration helpers."""
import io
import math
import statistics
import time

SERVICE_MODES = (
    "uncached", "cached_feature", "cached_coordinates", "cached_materialized_adapter"
)


def summarize(values):
    values = list(values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("expected nonnegative finite measurements")
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    low, high = math.floor(position), math.ceil(position)
    p90 = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "n": len(values), "median": statistics.median(values), "p90": p90,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values), "max": max(values),
    }


def measure(torch, fn, *, idle_check, warmup=10, repeats=50, cuda_events=True, device=None):
    """Time one resident callback on a caller-verified isolated CUDA device.

    idle_check is the caller's device/process-isolation check; it must raise
    if another process appears. It runs outside the measured callback. The
    caller supplies an already-restored evaluation model and fixed microbatch.
    Callbacks use the current stream or join their auxiliary work before returning.
    CPU-only stages still synchronize CUDA before and after the callback.
    Distinct stage measurements are standalone and must not be summed.
    """
    if (type(warmup) is not int or type(repeats) is not int
            or warmup < 10 or repeats < 30):
        raise ValueError("require at least 10 warmups and 30 measurements")
    if not callable(fn) or not callable(idle_check):
        raise ValueError("callback and device-isolation check are required")
    if not torch.cuda.is_initialized():
        raise ValueError("the caller must initialize its own CUDA context")
    device = torch.cuda.current_device() if device is None else device
    with torch.cuda.device(device), torch.inference_mode():
        idle_check()
        for _ in range(warmup):
            value = fn()
            del value
        torch.cuda.synchronize(device)
        idle_check()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        wall, cuda = [], []
        for index in range(repeats):
            if index and index % 10 == 0:
                idle_check()
            torch.cuda.synchronize(device)
            if cuda_events:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
            start = time.perf_counter()
            value = fn()
            if cuda_events:
                end.record()
                end.synchronize()
                cuda.append(float(begin.elapsed_time(end)))
            else:
                torch.cuda.synchronize(device)
            wall.append((time.perf_counter() - start) * 1000)
            del value
        torch.cuda.synchronize(device)
        idle_check()
        peak = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "wall_ms": summarize(wall),
        "cuda_stream_elapsed_ms": summarize(cuda) if cuda else None,
        "raw_wall_ms": wall, "raw_cuda_stream_elapsed_ms": cuda,
        "warmup": warmup, "repeats": repeats,
        "allocated_baseline_bytes": baseline, "reserved_baseline_bytes": reserved,
        "peak_allocated_bytes": peak, "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": peak - baseline,
        "peak_reserved_delta_bytes": peak_reserved - reserved,
    }


def profile_services(torch, callbacks, *, idle_check, warmup=10, repeats=50, device=None):
    """Measure the four cache modes on one identical native query microbatch."""
    if set(callbacks) != set(SERVICE_MODES):
        raise ValueError("provide exactly the four reported service callbacks")
    return {
        mode: measure(torch, callbacks[mode], idle_check=idle_check,
                      warmup=warmup, repeats=repeats, device=device)
        for mode in SERVICE_MODES
    }


def tensor_storage(torch, value):
    """Logical/unique bytes for one in-memory cache; do not write tensor files."""
    tensors = []

    def walk(item):
        if torch.is_tensor(item):
            tensors.append(item)
        elif isinstance(item, dict):
            for member in item.values():
                walk(member)
        elif isinstance(item, (tuple, list)):
            for member in item:
                walk(member)

    walk(value)
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        storages[key] = storage.nbytes()
    stream = io.BytesIO()
    torch.save(value, stream)
    serialized = stream.getbuffer().nbytes
    stream.close()
    return {
        "tensor_count": len(tensors),
        "logical_tensor_bytes": sum(t.numel() * t.element_size() for t in tensors),
        "unique_tensor_storage_bytes": sum(storages.values()),
        "persisted_task_artifact_bytes": 0,
        "torch_save_in_memory_serialized_bytes": serialized,
    }
