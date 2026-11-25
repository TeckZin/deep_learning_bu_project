"""Position codes shared by the refiner modules."""
import torch


def _grid(h, w, device=None, offset=None):
    """Row and column of every cell of an h x w grid, row-major: [h*w] each, or [B, h*w] with offset."""
    rows = torch.arange(h, device=device, dtype=torch.float32).repeat_interleave(w)   # [h*w]
    cols = torch.arange(w, device=device, dtype=torch.float32).repeat(h)               # [h*w]
    if offset is not None:
        offset = offset.to(device=device, dtype=torch.float32)
        rows = rows[None] + offset[:, :1]                                              # [B, h*w]
        cols = cols[None] + offset[:, 1:]
    return rows, cols


def pos2d(h, w, d, device=None, offset=None):
    """Fixed 2D sinusoidal position code for an h x w grid, row-major.

    offset=None      -> [h*w, d]      (grid starts at row 0, col 0: a full tile)
    offset [B, 2]    -> [B, h*w, d]   (row, col) of this grid's top-left cell
                                       inside the full tile, in grid cells.
    train.py passes the crop position here, so a 512 crop that came from the
    bottom-right of a tile gets the same codes it will have at full-tile inference.
    First d/2 channels encode the column, last d/2 the row.
    """
    assert d % 4 == 0, "d must be divisible by 4"
    rows, cols = _grid(h, w, device, offset)
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device, dtype=torch.float32) / (d // 4)))
    r = rows[..., None] * omega
    c = cols[..., None] * omega
    return torch.cat([c.sin(), c.cos(), r.sin(), r.cos()], dim=-1)


def rope2d(h, w, head_dim, device=None, offset=None, base=100.0):
    """cos, sin for 2D rotary position embedding on an h x w grid: each [1 or B, 1, h*w, head_dim].

    The first half of each head's dims rotates with the row, the second half with the column, so the
    attention score between two patches depends on their relative offset (Δrow, Δcol).
    base=100 (not 10000): the grid is at most 64 cells, so even the slowest frequency turns
    about 1 radian across a full tile. offset works as in pos2d (crop position in grid cells).
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4"
    rows, cols = _grid(h, w, device, offset)
    q = head_dim // 4
    freq = 1.0 / base ** (torch.arange(q, device=device, dtype=torch.float32) / q)
    r, c = rows[..., None] * freq, cols[..., None] * freq                # [(B,) h*w, q]
    angles = torch.cat([r, r, c, c], dim=-1)                              # [(B,) h*w, head_dim]
    angles = angles[None, None] if offset is None else angles[:, None]   # broadcast over heads
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate x [B, heads, N, head_dim] by the angles from rope2d (rotate-half form, per axis)."""
    r1, r2, c1, c2 = x.chunk(4, dim=-1)
    rotated = torch.cat([-r2, r1, -c2, c1], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)
