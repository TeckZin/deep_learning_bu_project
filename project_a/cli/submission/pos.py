"""Position codes shared by the refiner modules."""
import torch


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
    rows = torch.arange(h, device=device, dtype=torch.float32).repeat_interleave(w)   # [h*w]
    cols = torch.arange(w, device=device, dtype=torch.float32).repeat(h)               # [h*w]
    if offset is not None:
        offset = offset.to(device=device, dtype=torch.float32)
        rows = rows[None] + offset[:, :1]                                              # [B, h*w]
        cols = cols[None] + offset[:, 1:]
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device, dtype=torch.float32) / (d // 4)))
    r = rows[..., None] * omega
    c = cols[..., None] * omega
    return torch.cat([c.sin(), c.cos(), r.sin(), r.cos()], dim=-1)
