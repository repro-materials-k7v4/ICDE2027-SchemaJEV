"""Fixed-checkpoint interventions on native, document-level rank-64 factors."""
import math

STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
COMPONENTS = ("backbone", "encoder_only", "head_only", "full")
ENCODER_TARGETS = {
    "attn.Wqkv": ((768, 64), (64, 2304)),
    "attn.Wo": ((768, 64), (64, 768)),
    "mlp.Wi": ((768, 64), (64, 2304)),
    "mlp.Wo": ((1152, 64), (64, 768)),
}
DECISION_TARGETS = {
    "self_attn.in_proj_weight": ((768, 64), (64, 2304)),
    "self_attn.out_proj": ((768, 64), (64, 768)),
    "linear1": ((768, 64), (64, 3072)),
    "linear2": ((3072, 64), (64, 768)),
}


def validate_adapters(torch, adapters):
    """Validate all 96 sites after extracting one document's unbatched factors."""
    if not isinstance(adapters, dict) or set(adapters) != set(range(24)):
        raise ValueError("expected the complete 24-layer native factor dictionary")
    for layer in range(24):
        expected = ENCODER_TARGETS if layer < 22 else DECISION_TARGETS
        row = adapters[layer]
        if not isinstance(row, dict) or set(row) != set(expected):
            raise ValueError(f"invalid native targets at layer {layer}")
        for target, shapes in expected.items():
            pair = row[target]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("expected an A/B tensor pair")
            a, b = pair
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise ValueError("factors must be tensors")
            if tuple(a.shape) != shapes[0] or tuple(b.shape) != shapes[1]:
                raise ValueError(f"invalid factor shapes at {layer}:{target}")
            if a.dtype != b.dtype or a.device != b.device:
                raise ValueError("factor dtype and device must match")
            if not torch.is_floating_point(a) or not torch.is_floating_point(b):
                raise ValueError("floating factors required")
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                raise ValueError("non-finite factor; do not suppress it with zero strength")
    return 96


def transform_adapters(torch, adapters, *, strength=1.0, components="full"):
    """Retain A and multiply enabled B by strength; zero disabled B exactly.

    Native factors have A[in,64] and B[64,out], with unit adapter scaling.
    This changes the effective update to strength*(A@B); it does not change
    checkpoint parameters, native alpha/r, KD, or training balance lambda.
    The full/strength=1 path returns the original dictionary by identity.
    """
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or float(strength) not in STRENGTHS):
        raise ValueError("strength must belong to the reported fixed grid")
    if components not in COMPONENTS:
        raise ValueError("unknown component mode")
    validate_adapters(torch, adapters)
    strength = float(strength)
    if components == "full" and strength == 1.0:
        return adapters
    result = {}
    for layer, row in adapters.items():
        enabled = (components == "full" or
                   (components == "encoder_only" and layer < 22) or
                   (components == "head_only" and layer >= 22))
        result[layer] = {}
        for target, (a, b) in row.items():
            if not enabled or strength == 0.0:
                new_b = torch.zeros_like(b)
            else:
                new_b = b if strength == 1.0 else b * strength
            if not bool(torch.isfinite(new_b).all()):
                raise ValueError("non-finite transformed B")
            result[layer][target] = (a, new_b)
    return result
