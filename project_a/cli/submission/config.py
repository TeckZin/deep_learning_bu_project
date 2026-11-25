"""Refiner architecture config: one dataclass per module, named presets, and overrides.

Every architecture we train is a preset here. Changing a field changes the model
(and usually the checkpoint layout), so add a NEW preset instead of editing one that
has trained weights.

    build_refiner(c)                     -> PRESETS[ACTIVE]   (what predict.py / infer.py use)
    build_refiner(c, get_config("attempt6"))
    apply_overrides(cfg, ["core.layers=6", "tokenizer.fuse=concat"])   (ablations)

train.py stores the resolved config in the checkpoint metadata, so a checkpoint can
always be rebuilt with config_from_metadata(), whatever ACTIVE says.
"""
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import json


@dataclass(frozen=True)
class TokenizerConfig:
    """Doc stages 5, 7, 8: patchify features f and answer y_t (and memory z_t), fuse into one token per patch."""
    kind: str = "fused"           # "fused" (f, y) | "split" (x, y, z each tokenized, then fused; attempt10)
    fuse: str = "sum"             # fused: "sum" | "concat";  split: "concat" | "mlp"
    feat_norm: bool = True        # fused only: LayerNorm over the frozen feature channels first (attempt7)
    uncertainty: bool = True      # fused only: add a 4p(1-p) channel to the answer patches (attempt7)


@dataclass(frozen=True)
class MemoryConfig:
    """Doc stage 6: how z_0 is made and what is carried between steps."""
    kind: str = "slots"           # "slots": nz learned tokens | "spatial": Cz maps at H/stride (attempt10)
    nz: int = 32                  # slots only
    channels: int = 16            # spatial only: Cz
    stride: int = 4               # spatial only: z is [B, Cz, H/stride, W/stride]


@dataclass(frozen=True)
class CoreConfig:
    """Doc stages 9-11: the ONE shared transformer (QKV, attention, FFN, norms, layers)."""
    mode: str = "self"            # "self" (all tokens attend) | "cross" (grid <-> memory only)
    pos: str = "add"              # "add" (sinusoidal on tokens) | "rope" (2D RoPE on Q, K) | "both"
    layers: int = 4
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.05


@dataclass(frozen=True)
class HeadConfig:
    """Doc stages 12-13: grid tokens -> full-resolution correction dy (and z_t+1 for spatial memory)."""
    kind: str = "pixelshuffle"    # "pixelshuffle" | "spatial" (also sees the image, writes z; attempt10)
    hidden: int = 64
    dilations: tuple = (1, 2, 4, 8, 1)
    ctx: int = 16                 # core-output channels fed to the full-res conv stack (0 = off)
    gate: bool = True             # dy = (coarse + r) * sigmoid(g + 2)  (attempt7)


@dataclass(frozen=True)
class ImageConfig:
    """The full-resolution image inside the loop (TA ruling 2026-11-18), packed into f (submission/image.py)."""
    use: bool = False
    stride: int = 16              # packing stride = the feature grid stride (H/16)
    hidden: int = 24              # channels of the full-resolution image CNN
    layers: int = 3


@dataclass(frozen=True)
class RefinerConfig:
    d: int = 256
    patch: int = 16
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    image: ImageConfig = field(default_factory=ImageConfig)


PRESETS = {
    # attempt7 (git tag attempt7): the defaults above.
    "attempt7": RefinerConfig(),
    # attempt6: attempt7 without its tokenizer and head changes.
    "attempt6": RefinerConfig(
        tokenizer=TokenizerConfig(feat_norm=False, uncertainty=False),
        head=HeadConfig(ctx=0, gate=False)),
    # attempt10 (docs/experiments/attempt10_design.md): 6-layer encoder core with 2D RoPE over fused
    # (x, y, z) tokens, the image in a CNN inside the loop, and z as 16 maps at H/4.
    "attempt10": RefinerConfig(
        tokenizer=TokenizerConfig(kind="split", fuse="concat"),
        memory=MemoryConfig(kind="spatial", channels=16, stride=4),
        core=CoreConfig(pos="rope", layers=6, heads=8, ff_mult=4),
        head=HeadConfig(kind="spatial"),
        image=ImageConfig(use=True)),
}

# The preset build_refiner() uses when no config is given (staff predict.py, and
# infer.py for checkpoints without stored config). Set this to the architecture you submit.
# With no config, build_refiner() turns image.use off: the staff runner's base never packs the
# image into f. Our submission (infer.py) always passes the checkpoint's stored config.
ACTIVE = "attempt10"


def get_config(name=None):
    name = name or ACTIVE
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]


def _parse_value(text, current):
    if isinstance(current, bool):
        if text.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"Expected true/false, got {text!r}")
        return text.lower() in ("true", "1")
    if isinstance(current, tuple):
        return tuple(int(x) for x in text.strip("()[]").split(",") if x.strip())
    return type(current)(text)


def apply_overrides(cfg, overrides):
    """overrides: iterable of "section.field=value" (e.g. core.layers=6) or "field=value" (e.g. d=192)."""
    for item in overrides or ():
        key, sep, text = item.partition("=")
        path = key.strip().split(".")
        if not sep or len(path) > 2:
            raise ValueError(f"Override must look like section.field=value, got {item!r}")
        if len(path) == 1:
            if not hasattr(cfg, path[0]) or is_dataclass(getattr(cfg, path[0])):
                raise KeyError(f"Unknown config field {key!r}")
            cfg = replace(cfg, **{path[0]: _parse_value(text.strip(), getattr(cfg, path[0]))})
        else:
            section = getattr(cfg, path[0], None)
            if not is_dataclass(section) or path[1] not in {f.name for f in fields(section)}:
                raise KeyError(f"Unknown config field {key!r}")
            value = _parse_value(text.strip(), getattr(section, path[1]))
            cfg = replace(cfg, **{path[0]: replace(section, **{path[1]: value})})
    return cfg


def config_to_json(cfg):
    return json.dumps(asdict(cfg), sort_keys=True)


def config_from_json(text):
    raw = json.loads(text)
    sections = {"tokenizer": TokenizerConfig, "memory": MemoryConfig, "core": CoreConfig,
                "head": HeadConfig, "image": ImageConfig}
    kwargs = {}
    for key, value in raw.items():
        if key in sections:
            value = sections[key](**{k: tuple(v) if isinstance(v, list) else v
                                     for k, v in value.items()})
        kwargs[key] = value
    return RefinerConfig(**kwargs)


def config_from_metadata(metadata):
    """Config stored by train.py in a checkpoint's metadata, or None for older checkpoints."""
    text = (metadata or {}).get("refiner_cfg")
    return config_from_json(text) if text else None
