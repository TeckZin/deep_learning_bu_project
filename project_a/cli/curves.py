"""Training curves of one run as a small gallery for the repo: SVG charts + README.md (GitHub shows both).

    python curves.py --history weights/fdzeolla-attempt10/fold0/history.csv \
        --out ../../docs/experiments/attempt10_curves --title "attempt10, fold 0"

Reads train.py's history.csv (one row per epoch; validation and loop diagnostics only on validation
epochs) and writes, in --out:
  quality.svg    validation quality at K=0 (H0), 1, 2, 4
  metrics.svg    IoU, clDice and Boundary IoU at K=0 and K=4
  loss.svg       training loss per step, validation loss per K
  optim.svg      gradient norm per module, clip rate, learning rate
  loop.svg       per-step loop diagnostics (loss, |dy|, |dz|, GRU gate) at the first and last validation
  README.md      the charts, with the final-epoch numbers as tables
Columns missing from an older history.csv are skipped. No plotting library needed: plain SVG text.
"""
import argparse
import csv
import math
from html import escape
from pathlib import Path

KS = (0, 1, 2, 4)
# Colors: ordered things (K, step t) on one blue ramp, light -> dark; H0 / untrained as gray reference;
# categorical slots (fixed order) for unordered series such as modules.
INK, INK2, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"
REF = "#8c8c8c"
BLUE = {1: "#84b9f3", 2: "#338ced", 3: "#2370cd", 4: "#0e3d7e"}
CATEGORICAL = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
PANEL_W, PANEL_H = 420, 250
PAD_L, PAD_R, PAD_T, PAD_B = 52, 64, 34, 36      # right pad leaves room for end-of-line labels
FONT = "font-family='system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'"


# ---- data ----

def read_history(path):
    """{column: [(epoch, value), ...]} with empty cells dropped."""
    with Path(path).open() as fh:
        rows = list(csv.DictReader(fh))
    columns = {}
    for r in rows:
        epoch = int(r["epoch"])
        for k, v in r.items():
            if k != "epoch" and v not in ("", None):
                columns.setdefault(k, []).append((epoch, float(v)))
    return columns


def last_with(columns, prefix):
    """Epoch of the last row that has a column starting with `prefix`, or None."""
    epochs = [e for k, pts in columns.items() if k.startswith(prefix) for e, _ in pts]
    return max(epochs) if epochs else None


def first_with(columns, prefix):
    epochs = [e for k, pts in columns.items() if k.startswith(prefix) for e, _ in pts]
    return min(epochs) if epochs else None


def at(columns, key, epoch):
    return dict(columns.get(key, [])).get(epoch)


# ---- svg ----

def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + (abs(lo) or 1) * 0.1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi:
        ticks.append(ticks[-1] + step)
    return ticks


def fmt(v):
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if abs(v) >= 0.01:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{v:.1e}"


def spread(positions, gap=13, lo=None, hi=None):
    """Nudge label y positions apart (keeps order) so end labels never overlap."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    ys = [positions[i] for i in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if hi is not None and ys and ys[-1] > hi:
        shift = ys[-1] - hi
        ys = [y - shift for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    out = [0] * len(positions)
    for i, y in zip(order, ys):
        out[i] = y
    return out


def panel(title, series, x_label, y_label="", x_ticks=None, y_min=None, y_max=None):
    """One chart as SVG elements at origin (0, 0), PANEL_W x PANEL_H.

    series: dicts with label, points [(x, y)], color, and optional dash, markers, end_label.
    """
    series = [s for s in series if s["points"]]
    unit = f"<tspan font-weight='400' fill='{INK2}'>  ({escape(y_label)})</tspan>" if y_label else ""
    els = [f"<text x='0' y='14' font-size='13' font-weight='600' fill='{INK}' {FONT}>{escape(title)}{unit}</text>"]
    if not series:
        els.append(f"<text x='0' y='40' font-size='12' fill='{INK2}' {FONT}>no data in history.csv</text>")
        return els
    xs = [x for s in series for x, _ in s["points"]]
    ys = [y for s in series for _, y in s["points"]]
    lo = min(ys) if y_min is None else y_min
    hi = max(ys) if y_max is None else y_max
    margin = (hi - lo) * 0.06 or abs(hi) * 0.05 or 1
    bottom = lo if y_min is not None else lo - margin
    if lo >= 0 and bottom < 0:          # never pad below zero for data that cannot be negative
        bottom = 0
    yt = nice_ticks(bottom, hi + (0 if y_max is not None else margin))
    x0, x1 = min(xs), max(xs)
    xt = x_ticks or nice_ticks(x0, x1, 6)
    x0, x1 = min(x0, xt[0]), max(x1, xt[-1])
    w, h = PANEL_W - PAD_L - PAD_R, PANEL_H - PAD_T - PAD_B

    def sx(x):
        return PAD_L + (x - x0) / ((x1 - x0) or 1) * w

    def sy(y):
        return PAD_T + h - (y - yt[0]) / ((yt[-1] - yt[0]) or 1) * h

    for t in yt:
        y = sy(t)
        els.append(f"<line x1='{PAD_L}' x2='{PAD_L + w}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}' stroke-width='1'/>")
        els.append(f"<text x='{PAD_L - 6}' y='{y + 4:.1f}' font-size='11' fill='{INK2}' text-anchor='end' {FONT}>{fmt(t)}</text>")
    for t in xt:
        x = sx(t)
        els.append(f"<text x='{x:.1f}' y='{PAD_T + h + 16}' font-size='11' fill='{INK2}' text-anchor='middle' {FONT}>{fmt(t)}</text>")
    els.append(f"<line x1='{PAD_L}' x2='{PAD_L + w}' y1='{PAD_T + h}' y2='{PAD_T + h}' stroke='{INK2}' stroke-width='1'/>")
    els.append(f"<text x='{PAD_L + w / 2:.1f}' y='{PANEL_H - 4}' font-size='11' fill='{INK2}' text-anchor='middle' {FONT}>{escape(x_label)}</text>")

    ends = []
    for s in series:
        pts = sorted(s["points"])
        path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = f" stroke-dasharray='{s['dash']}'" if s.get("dash") else ""
        els.append(f"<path d='{path}' fill='none' stroke='{s['color']}' stroke-width='2' "
                   f"stroke-linejoin='round' stroke-linecap='round'{dash}/>")
        for x, y in pts if s.get("markers", len(pts) <= 12) else []:
            els.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' fill='{s['color']}' stroke='{SURFACE}' "
                       f"stroke-width='2'><title>{escape(s['label'])}: {fmt(y)} at {x_label} {fmt(x)}</title></circle>")
        if s.get("end_label", True):
            ends.append((s, sy(pts[-1][1]), sx(pts[-1][0])))
    label_ys = spread([y for _, y, _ in ends], lo=PAD_T, hi=PAD_T + h)
    for (s, _, x), y in zip(ends, label_ys):
        els.append(f"<text x='{x + 8:.1f}' y='{y + 4:.1f}' font-size='11' fill='{INK}' {FONT}>{escape(s['short'] if 'short' in s else s['label'])}</text>")
    return els


def legend(series, width):
    """One row of swatches + labels (always present for >= 2 series)."""
    els, x = [], 0
    for s in series:
        dash = f" stroke-dasharray='{s['dash']}'" if s.get("dash") else ""
        els.append(f"<line x1='{x}' x2='{x + 22}' y1='8' y2='8' stroke='{s['color']}' stroke-width='2'{dash}/>")
        els.append(f"<text x='{x + 28}' y='12' font-size='12' fill='{INK}' {FONT}>{escape(s['label'])}</text>")
        x += 28 + 7 * len(s["label"]) + 22
        if x > width - 80:
            break
    return els


def figure(path, title, panels, cols=2, legend_series=None):
    """Write panels ([elements]) on a grid with a title and an optional shared legend."""
    rows = (len(panels) + cols - 1) // cols
    head = 30 + (26 if legend_series else 0)
    W, H = cols * PANEL_W + (cols - 1) * 16, head + rows * PANEL_H + (rows - 1) * 12 + 8
    out = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>",
           f"<title>{escape(title)}</title>",
           f"<rect width='100%' height='100%' fill='{SURFACE}'/>",
           f"<text x='0' y='20' font-size='15' font-weight='600' fill='{INK}' {FONT}>{escape(title)}</text>"]
    if legend_series:
        out.append("<g transform='translate(0, 32)'>" + "".join(legend(legend_series, W)) + "</g>")
    for i, els in enumerate(panels):
        x, y = (i % cols) * (PANEL_W + 16), head + (i // cols) * (PANEL_H + 12)
        out.append(f"<g transform='translate({x}, {y})'>" + "".join(els) + "</g>")
    out.append("</svg>")
    Path(path).write_text("\n".join(out))


# ---- charts ----

def k_series(columns, key, scale=1.0):
    """K=0 as the gray H0 reference, K>=1 on the blue ramp."""
    out = []
    for k in KS:
        pts = [(e, v * scale) for e, v in columns.get(key.format(k=k), [])]
        if k == 0:
            out.append({"label": "K=0 (H0)", "short": "H0", "points": pts, "color": REF, "dash": "5 4"})
        else:
            out.append({"label": f"K={k}", "points": pts, "color": BLUE[k]})
    return out


def charts(columns, out, title):
    made = []
    epoch_ticks = None

    q = k_series(columns, "val_quality_k{k}")
    figure(out / "quality.svg", f"{title}: validation quality", [
        panel("Quality (0.5 IoU + 0.3 clDice + 0.2 BIoU)", q, "epoch", "percent", epoch_ticks)],
        cols=1, legend_series=q)
    made.append("quality")

    metric_panels = []
    for key, name in (("iou", "Road IoU"), ("cldice", "clDice"), ("biou", "Boundary IoU")):
        ser = [s for s in k_series(columns, f"val_{key}_k{{k}}") if s["label"] in ("K=0 (H0)", "K=4")]
        metric_panels.append(panel(name, ser, "epoch", "percent"))
    figure(out / "metrics.svg", f"{title}: validation metrics, H0 vs K=4", metric_panels, cols=3,
           legend_series=[s for s in k_series(columns, "val_iou_k{k}") if s["label"] in ("K=0 (H0)", "K=4")])
    made.append("metrics")

    train = [{"label": "y0 (H0)", "short": "y0", "points": columns.get("train_y0", []), "color": REF, "dash": "5 4"},
             {"label": "y0 after aug (input)", "short": "input", "points": columns.get("train_input", []),
              "color": REF, "dash": "1 4"}]
    # steps 2 and 3 lie on top of 1 and 4, so only the outer two get end labels (the legend names all)
    train += [{"label": f"step {t}", "short": f"y{t}", "points": columns.get(f"train_y{t}", []), "color": BLUE[t],
               "end_label": t in (1, 4)} for t in (1, 2, 3, 4)]
    val = k_series(columns, "val_loss_k{k}")
    figure(out / "loss.svg", f"{title}: loss", [
        panel("Training loss per step (epoch mean)", train, "epoch"),
        panel("Validation loss per K", val, "epoch")], cols=2,
        legend_series=[s for s in train if s["points"]])
    made.append("loss")

    modules = sorted(k[5:] for k in columns if k.startswith("grad_") and k != "grad_total")
    grads = [{"label": m, "points": columns[f"grad_{m}"], "color": CATEGORICAL[i % 8], "markers": False}
             for i, m in enumerate(modules)]
    grads.append({"label": "total", "points": columns.get("grad_total", []), "color": INK, "dash": "5 4",
                  "markers": False})
    clip = [{"label": "clipped", "points": [(e, 100 * v) for e, v in columns.get("clip_rate", [])],
             "color": CATEGORICAL[0], "end_label": False, "markers": False}]
    lr = [{"label": "lr", "points": columns.get("lr", []), "color": CATEGORICAL[0], "end_label": False,
           "markers": False}]
    figure(out / "optim.svg", f"{title}: optimization", [
        panel("Gradient norm by module (before clipping)", grads, "epoch"),
        panel("Iterations clipped", clip, "epoch", "percent", y_min=0),
        panel("Learning rate", lr, "epoch", y_min=0)], cols=2, legend_series=grads)
    made.append("optim")

    first, last = first_with(columns, "diag_loss_t"), last_with(columns, "diag_loss_t")
    if last is not None:
        steps = sorted(int(k.rsplit("_t", 1)[1]) for k in columns if k.startswith("diag_dy_t"))
        loop_panels = []
        for key, name, ts in (("loss", "Loss after step t (t=0 is H0)", [0] + steps), ("dy", "Mean |dy|", steps),
                              ("dz", "Mean |dz|", steps), ("gate", "GRU update gate u", steps)):
            ser = []
            for epoch, color, dash, label in ((first, REF, "5 4", f"epoch {first}"), (last, BLUE[4], None, f"epoch {last}")):
                if epoch is None or (epoch == first and first == last and color == REF):
                    continue
                pts = [(t, at(columns, f"diag_{key}_t{t}", epoch)) for t in ts]
                pts = [(t, v) for t, v in pts if v is not None]
                ser.append({"label": label, "points": pts, "color": color, "dash": dash, "markers": True})
            loop_panels.append(panel(name, ser, "step t", x_ticks=ts))
        figure(out / "loop.svg", f"{title}: loop diagnostics (first 50 validation tiles, 8 steps)", loop_panels,
               cols=2, legend_series=[{"label": f"epoch {first} (untrained)" if first == 0 else f"epoch {first}",
                                       "color": REF, "dash": "5 4"},
                                      {"label": f"epoch {last}", "color": BLUE[4]}])
        made.append("loop")
    return made


# ---- readme ----

def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def readme(columns, made, title):
    val_epochs = sorted({e for e, _ in columns.get("val_quality_k4", [])})
    last = val_epochs[-1] if val_epochs else None
    best = max(columns.get("val_quality_k4", []), key=lambda p: p[1], default=(None, None))[0]
    lines = [f"# Training curves: {title}", "",
             "Made by `project_a/cli/curves.py` from the run's `history.csv`. Validation runs on the whole "
             "held-out fold every few epochs (the points); training values are epoch means. "
             "Quality = 0.5 IoU + 0.3 clDice + 0.2 Boundary IoU, in percent, counted over all pixels of the fold.", ""]
    if last is not None:
        g = lambda k, e=last: at(columns, k, e)
        lines += [f"Last validation: epoch {last}. Best K=4 quality: epoch {best}.", ""]
        lines += table(["Epoch " + str(last), "K=0 (H0)", "K=1", "K=2", "K=4"],
                       [[name] + [fmt(g(f"val_{key}_k{k}")) if g(f"val_{key}_k{k}") is not None else "" for k in KS]
                        for key, name in (("quality", "Quality"), ("iou", "IoU"), ("cldice", "clDice"),
                                          ("biou", "Boundary IoU"), ("loss", "Loss"))])
        lines.append("")
    sections = {
        "quality": ("Validation quality", "Each K is the same refiner run for K steps. The gap between K=1 and "
                    "K=4 is what steps 2-4 add."),
        "metrics": ("Validation metrics", "The three parts of quality, H0 against K=4."),
        "loss": ("Loss", "Left: training loss after each unrolled step, with H0's loss and the loss of the "
                 "augmented y0 the refiner actually receives. Right: validation loss per K."),
        "optim": ("Optimization", "Gradient norm per module (epoch mean, before clipping), share of iterations "
                  "clipped, and the learning-rate schedule."),
        "loop": ("Loop diagnostics", "Per step t on the first 50 validation tiles, run for 8 steps (past the "
                 "trained K): loss after step t, mean size of the change to y and to z, and the mean GRU gate. "
                 "Gray is the first validation (untrained when it is epoch 0), blue the last."),
    }
    for name in made:
        head, text = sections[name]
        lines += [f"## {head}", "", text, "", f"![{head}]({name}.svg)", ""]
    if "loop" in made:
        e = last_with(columns, "diag_loss_t")
        steps = sorted(int(k.rsplit("_t", 1)[1]) for k in columns if k.startswith("diag_dy_t"))
        rows = [[name] + [fmt(at(columns, f"diag_{key}_t{t}", e)) if at(columns, f"diag_{key}_t{t}", e) is not None
                          else "" for t in steps]
                for key, name in (("loss", "Loss"), ("dy", "Mean abs dy"), ("dz", "Mean abs dz"), ("gate", "Gate u"))]
        lines += [f"Loop diagnostics at epoch {e}:", ""] + table(["Step t"] + [str(t) for t in steps], rows)
        extra = [(k, n) for k, n in (("diag_quality_k4", "K=4"), ("diag_quality_k8", "K=8"),
                                     ("diag_quality_k4_reset", "K=4, z reset every step"),
                                     ("diag_quality_k4_nofeedback", "K=4, y0 fed back instead of y_t"))
                 if at(columns, k, e) is not None]
        if extra:
            lines += ["", "Quality on the same 50 tiles:", ""]
            lines += table([n for _, n in extra], [[fmt(at(columns, k, e)) for k, _ in extra]])
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", type=Path, required=True, help="history.csv from train.py")
    p.add_argument("--out", type=Path, required=True, help="folder for the SVGs and README.md")
    p.add_argument("--title", help="e.g. 'attempt10, fold 0' (default: the history's folder name)")
    args = p.parse_args()
    columns = read_history(args.history)
    title = args.title or args.history.parent.name
    args.out.mkdir(parents=True, exist_ok=True)
    made = charts(columns, args.out, title)
    (args.out / "README.md").write_text(readme(columns, made, title))
    print(f"wrote {args.out}/README.md and {', '.join(m + '.svg' for m in made)}")


if __name__ == "__main__":
    main()
