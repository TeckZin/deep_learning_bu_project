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
    """Doc stages 5, 7, 8: patchify features f and answer y_t, fuse them into one token per patch."""
    kind: str = "fused"
    fuse: str = "sum"             # "sum" | "concat" (concat -> Linear(2d, d))
    feat_norm: bool = True        # LayerNorm over the frozen feature channels first (attempt7)
    uncertainty: bool = True      # add a 4p(1-p) channel to the answer patches (attempt7)


@dataclass(frozen=True)
class MemoryConfig:
    """Doc stage 6: how z_0 is made and what is carried between steps."""
    kind: str = "slots"           # learned memory slots that read the grid once
    nz: int = 32


@dataclass(frozen=True)
class CoreConfig:
    """Doc stages 9-11: the ONE shared transformer (QKV, attention, FFN, norms, layers)."""
    mode: str = "self"            # "self" (all tokens attend) | "cross" (grid <-> memory only)
    layers: int = 4
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.05


@dataclass(frozen=True)
class HeadConfig:
    """Doc stages 12-13: grid tokens -> full-resolution correction dy."""
    kind: str = "pixelshuffle"
    hidden: int = 64
    dilations: tuple = (1, 2, 4, 8, 1)
    ctx: int = 16                 # core-output channels fed to the full-res conv stack (0 = off)
    gate: bool = True             # dy = (coarse + r) * sigmoid(g + 2)  (attempt7)


@dataclass(frozen=True)
class RefinerConfig:
    d: int = 256
    patch: int = 16
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    head: HeadConfig = field(default_factory=HeadConfig)


PRESETS = {
    # attempt7 (git tag attempt7): the defaults above.
    "attempt7": RefinerConfig(),
    # attempt6: attempt7 without its tokenizer and head changes.
    "attempt6": RefinerConfig(
        tokenizer=TokenizerConfig(feat_norm=False, uncertainty=False),
        head=HeadConfig(ctx=0, gate=False)),
}

# The preset build_refiner() uses when no config is given (staff predict.py, and
# infer.py for checkpoints without stored config). Set this to the architecture you submit.
ACTIVE = "attempt7"


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
                "head": HeadConfig}
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
