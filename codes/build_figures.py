#!/usr/bin/env python3

import csv
import gzip
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
STRUCT = os.path.join(HERE, "..", "structures")

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#b9b8b3"
WHITE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"

TITLE_SIZE = 9.5
TITLE_PAD = 7

plt.rcParams.update({
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": "#e9e8e4", "grid.linewidth": 0.5,
    "lines.linewidth": 2.0, "lines.markersize": 4.5,
})


def tidy(ax, ylab=None, xlab=None, title=None, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis=grid, which="major")
    if ylab:
        ax.set_ylabel(ylab)
    if xlab:
        ax.set_xlabel(xlab)
    if title:
        ax.set_title(title, loc="left", color=INK, pad=TITLE_PAD)


def panel_extent(ax, renderer):

    from matplotlib.transforms import Bbox
    boxes = [ax.get_tightbbox(renderer)]
    for lbl in (ax.yaxis.label, ax.xaxis.label):
        if lbl.get_text().strip():
            boxes.append(lbl.get_window_extent(renderer))
    for axis in ("y", "x"):
        for lbl in drawn_ticklabels(ax, axis):
            boxes.append(lbl.get_window_extent(renderer))
    return Bbox.union([b for b in boxes if b.width and b.height])


def panel_headings(fig):

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    gap = (TITLE_PAD / 72.0) / fig.get_figheight()
    moved = []
    for ax in fig.axes:
        t = ax.get_title(loc="left")
        if not t:
            continue
        ax.set_title("", loc="left")
        moved.append((ax, t))
    if not moved:
        return


    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = {ax: inv.transform_bbox(panel_extent(ax, r)) for ax, _ in moved}


    rows_ = {}
    for ax, _ in moved:
        key = round(ax.get_position().y1, 3)
        rows_[key] = max(rows_.get(key, 0.0), box[ax].y1)


    cols = {}
    for ax, _ in moved:
        key = round(ax.get_position().x0, 3)
        cols[key] = min(cols.get(key, 1.0), box[ax].x0)


    left_of = {}
    for key in rows_:
        row = sorted((a for a, _ in moved
                      if round(a.get_position().y1, 3) == key),
                     key=lambda a: a.get_position().x0)
        for prev, ax in zip(row, row[1:]):
            left_of[ax] = box[prev].x1
    placed = {}
    for ax, t in moved:
        y = rows_[round(ax.get_position().y1, 3)] + gap
        x = cols[round(ax.get_position().x0, 3)]
        if ax in left_of:
            x = max(x, (left_of[ax] + ax.get_position().x0) / 2.0)
        placed[ax] = fig.text(x, y, t, ha="left", va="bottom",
                              color=INK, fontsize=TITLE_SIZE, fontweight="bold")


    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = (6.0 / 72.0) / fig.get_figwidth()
    for key in rows_:
        row = sorted((a for a, _ in moved
                      if round(a.get_position().y1, 3) == key),
                     key=lambda a: a.get_position().x0)
        right = None
        for ax in row:
            txt = placed[ax]
            ext = inv.transform_bbox(txt.get_window_extent(r))
            if right is not None and ext.x0 < right + pad:


                txt.set_x(ax.get_position().x0)
                fig.canvas.draw()
                ext = inv.transform_bbox(txt.get_window_extent(r))
                if ext.x0 < right + pad:
                    txt.set_x(right + pad)
                    fig.canvas.draw()
                    ext = inv.transform_bbox(txt.get_window_extent(r))
            right = ext.x1


def drawn_ticklabels(ax, axis):

    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    lo, hi = min(lo, hi), max(lo, hi)
    locs = ax.get_yticks() if axis == "y" else ax.get_xticks()
    labs = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    return [l for loc, l in zip(locs, labs)
            if lo - 1e-9 <= loc <= hi + 1e-9 and l.get_text().strip()]


def check_headings(fig, name):

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    heads = [(t.get_text(), t.get_window_extent(r)) for t in fig.texts]
    others = []
    for ax in fig.axes:
        items = (list(ax.texts) + drawn_ticklabels(ax, "y")
                 + drawn_ticklabels(ax, "x"))
        items += [l for l in (ax.yaxis.label, ax.xaxis.label)
                  if l.get_text().strip()]
        lg = ax.get_legend()
        if lg is not None:
            items += list(lg.get_texts())
        others += [(t.get_text(), t.get_window_extent(r))
                   for t in items if t.get_text().strip()]
    bad = []
    for hs, he in heads:
        bad += ["%s: heading %r lands on %r" % (name, hs, o)
                for o, oe in others if he.overlaps(oe)]
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            if heads[i][1].overlaps(heads[j][1]):
                bad.append("%s: heading %r lands on heading %r"
                           % (name, heads[i][0], heads[j][0]))


    for ax in fig.axes:
        lab = [t for t in ax.texts if t.get_text().strip()]
        for i in range(len(lab)):
            for j in range(i + 1, len(lab)):
                if lab[i].get_window_extent(r).overlaps(
                        lab[j].get_window_extent(r)):
                    bad.append("%s: label %r lands on label %r"
                               % (name, lab[i].get_text(), lab[j].get_text()))


    from matplotlib.transforms import Bbox
    drawn = {}
    for n, ax in enumerate(fig.axes):
        bs = []
        for art in list(ax.lines) + list(ax.collections) + list(ax.patches):
            if not art.get_visible():
                continue
            try:
                e = art.get_window_extent(r)
            except Exception:
                continue


            e = Bbox.intersection(e, ax.get_window_extent(r))
            if e is not None and e.width and e.height:
                bs.append(e)
        drawn[n] = bs
    for n, ax in enumerate(fig.axes):
        items = ([t for t in ax.texts if t.get_text().strip()]
                 + drawn_ticklabels(ax, "y") + drawn_ticklabels(ax, "x"))
        for t in items:
            e = t.get_window_extent(r)
            for m, marks in drawn.items():
                if m == n:
                    continue
                if any(e.overlaps(b) for b in marks):
                    bad.append("%s: %r from one panel lands on another panel's "
                               "drawing" % (name, t.get_text()))
                    break
    if bad:
        raise SystemExit("\n  ".join(["fatal: overlapping text in %s" % name] + bad))


def save(fig, name):
    panel_headings(fig)
    check_headings(fig, name)
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=400, bbox_inches="tight",
                facecolor=WHITE)
    fig.savefig(os.path.join(FIGS, name + ".svg"), bbox_inches="tight",
                facecolor=WHITE)
    plt.close(fig)
    print("wrote", name + ".png and .svg")


def rows(path):
    with open(os.path.join(DATA, path)) as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def avg_ranks(v):

    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):

    if len(xs) != len(ys):
        raise SystemExit("fatal: spearman on unequal lengths")
    rx, ry = avg_ranks(list(xs)), avg_ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return cov / den if den else float("nan")


def figure1():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 4.1))
    cw = ch = 1.0

    def block(ax, x0, letters, colour):
        for i, ltr in enumerate(letters):
            y = -i * ch
            fc = WHITE if ltr == "-" else colour
            ax.add_patch(Rectangle((x0, y), cw * 0.86, ch * 0.86, facecolor=fc,
                                   edgecolor=GREY, linewidth=0.5))
            ax.text(x0 + cw * 0.43, y + ch * 0.43, ltr, ha="center", va="center",
                    fontsize=7.5, weight="bold" if ltr != "-" else "normal",
                    color=WHITE if ltr != "-" else MUTED)

    axa.set_xlim(-0.9, 7.6)
    axa.set_ylim(-11.6, 4.3)
    axa.axis("off")
    axa.text(-0.85, 3.75, "a", fontsize=11, weight="bold", color=INK)
    axa.text(-0.2, 3.75, "Gap inflation", fontsize=9.5, weight="bold", color=INK,
             va="center")
    axa.text(-0.2, 2.75, "absent evidence scores as constraint", fontsize=8.5,
             color=INK2, va="center")


    def schematic(letters):
        n = len(letters)
        counts = {}
        for x in letters:
            counts[x] = counts.get(x, 0) + 1
        h = -sum(v / n * math.log2(v / n) for v in counts.values())
        h = abs(h)
        return h, 1.0 - h / math.log2(21)

    cols_a = [(S1, list("LLILVLLI"), "ordinary\ncolumn"),
              (GREY, ["-"] * 8, "all-gap\ncolumn")]
    for c, (col, letters, lab) in enumerate(cols_a):
        x0 = 1.1 + c * 2.7
        block(axa, x0, letters, col)
        axa.text(x0 + cw * 0.43, 0.95, lab, ha="center", va="bottom",
                 fontsize=8, color=INK2)
    axa.text(0.85, -3.6, "aligned\nhomologues", ha="right", va="center",
             fontsize=8, color=INK2)
    h0, c0 = schematic(cols_a[0][1])
    h1, c1 = schematic(cols_a[1][1])
    axa.text(1.53, -8.1, "H = %.2f bits\nreported %.2f" % (h0, c0),
             ha="center", va="top", fontsize=8, color=INK)
    axa.text(4.23, -8.1, "H = %.2f bits\nreported %.2f" % (h1, c1),
             ha="center", va="top", fontsize=8, color=S2, weight="bold")
    axa.add_patch(FancyArrowPatch((5.3, -3.6), (6.4, -3.6), arrowstyle="-|>",
                                  mutation_scale=11, color=S2, linewidth=1.4))
    axa.text(6.5, -3.6, "scored as\nmaximally\nconserved", ha="left", va="center",
             fontsize=8, color=S2)
    axa.text(-0.2, -10.1,
             "conservation = 1 − H / log₂ 21, the 21st symbol being the gap",
             fontsize=7.8, color=INK2, style="italic")

    axb.set_xlim(-2.7, 7.4)
    axb.set_ylim(-11.6, 4.3)
    axb.axis("off")
    axb.text(-2.65, 3.75, "b", fontsize=11, weight="bold", color=INK)
    axb.text(-2.0, 3.75, "Clade partitioning", fontsize=9.5, weight="bold",
             color=INK, va="center")
    axb.text(-2.0, 2.75, "divergence scores as tolerance", fontsize=8.5,
             color=INK2, va="center")
    clades = [("D", S1, "LPO"), ("R", S2, "EPX"), ("A", S3, "TPO"), ("M", S4, "MPO")]
    x0, y = 1.9, 0.0
    tipy = []
    for ltr, col, _ in clades:
        ys = []
        for _ in range(2):
            axb.add_patch(Rectangle((x0, y), cw * 0.86, ch * 0.86, facecolor=col,
                                    edgecolor=GREY, linewidth=0.5))
            axb.text(x0 + cw * 0.43, y + ch * 0.43, ltr, ha="center", va="center",
                     fontsize=7.5, color=WHITE, weight="bold")
            ys.append(y + ch * 0.43)
            y -= ch
        y -= 0.42
        tipy.append(float(np.mean(ys)))
    for ty, (_l, col, nm) in zip(tipy, clades):
        axb.plot([0.75, 1.8], [ty, ty], color=col, lw=2.0, solid_capstyle="round")
        axb.text(0.62, ty, nm, ha="right", va="center", fontsize=7.8, color=col,
                 weight="bold")
    axb.plot([0.75, 0.75], [tipy[0], tipy[1]], color=MUTED, lw=1.1)
    axb.plot([0.75, 0.75], [tipy[2], tipy[3]], color=MUTED, lw=1.1)
    m1, m2 = (tipy[0] + tipy[1]) / 2, (tipy[2] + tipy[3]) / 2
    axb.plot([0.35, 0.75], [m1, m1], color=MUTED, lw=1.1)
    axb.plot([0.35, 0.75], [m2, m2], color=MUTED, lw=1.1)
    axb.plot([0.35, 0.35], [m1, m2], color=MUTED, lw=1.1)
    axb.plot([0.0, 0.35], [(m1 + m2) / 2, (m1 + m2) / 2], color=MUTED, lw=1.1)
    axb.text(x0 + cw * 0.43, 0.95, "one column", ha="center", va="bottom",
             fontsize=8, color=INK2)
    axb.text(3.25, -1.6, "fixed within\nevery clade", ha="left", va="center",
             fontsize=8, color=INK2)
    axb.text(3.25, -7.0, "different\nbetween clades", ha="left", va="center",
             fontsize=8, color=INK2)
    drawn = "".join(l for l, _c, _n in clades for _ in range(2))
    nb = len(drawn)
    cb = {}
    for x in drawn:
        cb[x] = cb.get(x, 0) + 1
    hb = -sum(v / nb * math.log2(v / nb) for v in cb.values())
    fam_id = (drawn.count(clades[0][0]) - 1) / (nb - 1)
    axb.text(-2.0, -9.5,
             "family identity %.2f, %.2f bits\nown-clade identity 1.00"
             % (fam_id, hb), ha="left", va="top", fontsize=8, color=INK)
    axb.add_patch(FancyArrowPatch((5.3, -4.3), (6.3, -4.3), arrowstyle="-|>",
                                  mutation_scale=11, color=S2, linewidth=1.4))
    axb.text(6.4, -4.3, "scored as\nfreely\nvariable", ha="left", va="center",
             fontsize=8, color=S2)
    fig.tight_layout()
    save(fig, "Figure1")


def _catmull_rom(pts, n=6):

    pts = np.asarray(pts, dtype=float)
    m = len(pts)
    if m < 4:
        return pts, np.arange(max(m - 1, 0))
    out, src = [], []
    for i in range(m - 1):
        p0 = pts[max(i - 1, 0)]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[min(i + 2, m - 1)]
        for j in range(n):
            t = j / n
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(q)
            src.append(i)
    out.append(pts[-1])
    return np.asarray(out), np.asarray(src)


def figure4():
    pdb = os.path.join(STRUCT, "7DE5.pdb")
    mapfile = os.path.join(STRUCT, "7DE5_to_human.json")
    if not (os.path.exists(pdb) and os.path.exists(mapfile)):
        print("skipping Figure S4: structure or mapping missing")
        return
    mapping = {int(k): v for k, v in json.load(open(mapfile)).items()}
    ca = []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "A":
            rn = int(line[22:26])
            if ca and ca[-1][0] == rn:
                continue
            ca.append((rn, float(line[30:38]), float(line[38:46]),
                       float(line[46:54])))

    panel = {int(r["position"]): r for r in rows("reliability_panel_lpo.csv")}
    tn = {int(r["position"]): r for r in rows("tree_null_peroxidase.csv")}

    def verdict(hp):
        r = panel.get(hp)
        if r is None:
            return "unassessed"
        if r["verdict"] == "gap_inflated":
            return "gap_inflated"
        t = tn.get(hp)
        if t is not None and t.get("call_tree") == "diagnostic":
            return "clade_partitioned"
        if r["verdict"] == "clade_partitioned":
            return "indeterminate"
        return "ok"

    colmap = {"ok": "#b9b7b1", "clade_partitioned": S2, "gap_inflated": S4,
              "indeterminate": "#dedcd6", "unassessed": "#eeede9"}
    flagged = {"clade_partitioned", "gap_inflated"}

    xyz = np.array([[c[1], c[2], c[3]] for c in ca], dtype=float)
    verd = [verdict(mapping.get(c[0], -1)) for c in ca]


    cen = xyz.mean(axis=0)
    _u, _s, vt = np.linalg.svd(xyz - cen, full_matrices=False)
    proj = (xyz - cen) @ vt.T

    dense, src = _catmull_rom(proj, n=6)
    seg_depth = 0.5 * (dense[:-1, 2] + dense[1:, 2])
    lo, hi = float(seg_depth.min()), float(seg_depth.max())
    span = max(hi - lo, 1e-9)

    fig = plt.figure(figsize=(7.4, 3.9))


    gsx = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.52)
    ax = fig.add_subplot(gsx[0, 0])

    def draw(idx, extra_lw=0.0, on_top=False):
        for i in idx:
            j = src[i]
            d = (seg_depth[i] - lo) / span
            lw = (1.55 + 1.05 * d) + extra_lw
            seg = dense[i:i + 2]
            ax.plot(seg[:, 0], seg[:, 1], color=WHITE,
                    linewidth=lw + 1.5, solid_capstyle="round",
                    zorder=(4.0 if on_top else 2.0) + d)
            ax.plot(seg[:, 0], seg[:, 1], color=colmap[verd[j]],
                    linewidth=lw, solid_capstyle="round",
                    zorder=(4.05 if on_top else 2.05) + d)

    draw(np.argsort(seg_depth))
    draw([i for i in range(len(src)) if verd[src[i]] in flagged],
         extra_lw=0.45, on_top=True)

    inv = {v: k for k, v in mapping.items()}
    byres = {c[0]: i for i, c in enumerate(ca)}
    marks = [(144, "D144", 150.0), (205, "E205", 232.0),
             (313, "E313", 38.0), (648, "E648", 203.0)]
    for hp, lab, ang in marks:
        k = inv.get(hp)
        if k is None or k not in byres:
            continue
        q = proj[byres[k]]
        ax.scatter([q[0]], [q[1]], s=46, color=INK, edgecolor=WHITE,
                   linewidth=1.1, zorder=9)
        th = math.radians(ang)
        r = 17.0
        tx, ty = q[0] + r * math.cos(th), q[1] + r * math.sin(th)
        ax.plot([q[0], tx], [q[1], ty], color=INK, lw=0.8, zorder=8.5)
        ax.annotate(lab, (tx, ty), xytext=(3.2 * math.cos(th),
                                           3.2 * math.sin(th)),
                    textcoords="offset points", fontsize=7.6, color=INK,
                    weight="bold", zorder=10,
                    ha="left" if math.cos(th) > 0.15 else
                       ("right" if math.cos(th) < -0.15 else "center"),
                    va="bottom" if math.sin(th) > 0.15 else
                       ("top" if math.sin(th) < -0.15 else "center"))

    ax.set_aspect("equal", adjustable="datalim")
    hw = float(np.abs(proj[:, :2]).max()) + 21.0
    ax.set_xlim(-hw, hw)
    ax.set_ylim(-hw * 0.98, hw * 0.98)
    ax.axis("off")
    ax.set_facecolor(WHITE)
    ax.set_title("a  Lactoperoxidase fold by verdict", loc="left", color=INK,
                 fontsize=9.5, fontweight="bold", pad=7)
    handles = [Line2D([0], [0], color=colmap["ok"], lw=3, label="no flag"),
               Line2D([0], [0], color=colmap["clade_partitioned"], lw=3,
                      label="clade-partitioned"),
               Line2D([0], [0], color=colmap["gap_inflated"], lw=3,
                      label="gap-inflated"),
               Line2D([0], [0], color=colmap["indeterminate"], lw=3,
                      label="indeterminate")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=2, fontsize=7.0, handlelength=1.5, columnspacing=1.0,
              frameon=False)

    ax2 = fig.add_subplot(gsx[0, 1])
    order = ["ok", "indeterminate", "clade_partitioned", "gap_inflated"]
    lane = {k: i for i, k in enumerate(order)}
    for pos in sorted(panel):
        v = verdict(pos)
        if v == "unassessed":
            continue
        ax2.add_patch(Rectangle((pos - 0.5, lane[v] - 0.40), 1.0, 0.80,
                                facecolor=colmap[v], edgecolor="none"))
    dxs = {144: -18, 205: 18, 313: 0, 648: 0}
    for hp, lab, _ang in marks:
        ax2.annotate(lab, (hp, lane[verdict(hp)] + 0.42),
                     xytext=(dxs.get(hp, 0), 13), textcoords="offset points",
                     ha="center", fontsize=7.2, color=INK, weight="bold",
                     arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(["no flag", "indeterminate", "clade-partitioned",
                         "gap-inflated"], fontsize=7.8)
    ax2.set_ylim(-0.75, len(order) - 0.05)
    ax2.set_xlim(0, 722)
    tidy(ax2, None, "human lactoperoxidase position",
         "b  Verdicts along the sequence", grid="x")
    fig.tight_layout()
    save(fig, "Figure4")


def _gap_panels(ax1, ax2):
    with gzip.open(os.path.join(DATA, "gap_inflation_positions.csv.gz"), "rt") as fh:
        pos = list(csv.DictReader(fh))
    dec = defaultdict(lambda: {"rep": [], "free": []})
    for r in pos:
        g, c, u = num(r["gap_frac"]), num(r["conservation"]), num(r["cons_ungapped"])
        if None in (g, c, u):
            continue
        d = min(9, int(g * 10))
        dec[d]["rep"].append(c)
        dec[d]["free"].append(u)
    ds = sorted(dec)
    rep = [float(np.mean(dec[d]["rep"])) for d in ds]
    free = [float(np.mean(dec[d]["free"])) for d in ds]
    am = {}
    for r in rows("miscalibration_gap_summary.csv"):
        if r["measure"].startswith("mean_AM_gap_decile_"):
            am[int(r["measure"].rsplit("_", 1)[1])] = num(r["value"])
    amv = [am.get(d) for d in ds]

    x = np.arange(len(ds))
    ax1.plot(x, rep, color=S2, marker="o", zorder=3, label="conservation as reported")
    ax1.plot(x, free, color=S3, marker="s", zorder=3, label="gap-free recomputation")
    ax1.plot(x, amv, color=S1, marker="^", zorder=3, label="AlphaMissense")
    for xi in (0, len(x) - 1):
        ax1.annotate("%.2f" % rep[xi], (x[xi], rep[xi]), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=7.5,
                     color=S2, weight="bold")
        ax1.annotate("%.2f" % free[xi], (x[xi], free[xi]), xytext=(0, -14),
                     textcoords="offset points", ha="center", fontsize=7.5, color=S3)
        ax1.annotate("%.2f" % amv[xi], (x[xi], amv[xi]),
                     xytext=(0, 7 if xi == 0 else -14),
                     textcoords="offset points", ha="center", fontsize=7.5, color=S1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["%d" % (10 * d) for d in ds])
    ax1.set_ylim(0.2, 1.12)
    ax1.legend(loc="upper left")
    tidy(ax1, "mean score", "gap fraction of the column (%)",
         "a  Gappiness inflates the reported score")

    gi = rows("gap_inflation_summary.csv")
    pear = {r["cohort"]: num(r["value"]) for r in gi
            if r["measure"] == "pearson_gap_vs_conservation"}
    pear_u = {r["cohort"]: num(r["value"]) for r in gi
              if r["measure"] == "pearson_gap_vs_conservation_ungapped"}
    cohorts = ["data-human", "data-alphahelical", "both"]
    xx = np.arange(3)
    w = 0.36
    a = [pear.get(c) for c in cohorts]
    b = [pear_u.get(c) for c in cohorts]
    ax2.bar(xx - w / 2 - 0.012, a, w, color=S2, label="as reported")
    ax2.bar(xx + w / 2 + 0.012, b, w, color=S3, label="gap-free")
    for xi, v in zip(xx - w / 2 - 0.012, a):
        ax2.annotate("%+.2f" % v, (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5, color=INK)
    for xi, v in zip(xx + w / 2 + 0.012, b):
        ax2.annotate("%+.2f" % v, (xi, v), xytext=(0, 3 if v >= 0 else -11),
                     textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax2.axhline(0, color=MUTED, lw=0.7)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(["human", "alpha-helical", "both"])
    ax2.set_ylim(-0.13, 0.80)
    ax2.legend(loc="upper left")
    tidy(ax2, "\u03c1 (gap fraction, conservation)", None,
         "b  The association is a gap artefact")


def figure3():
    dl = rows("depth_ladder_summary.csv")
    depths = ["D1_paralogues", "D2_orthologues", "D3_subfamily", "D4_family"]
    nice = ["3\nparalogues", "5\northologues", "16\nsubfamily", "79\nfamily"]
    calls = [("conserved", S1, "o"), ("diagnostic", S2, "s"),
             ("tolerant", S3, "^"), ("indeterminate", S4, "D")]
    mi = defaultdict(dict)
    for r in dl:
        if r["measure"] == "mean_identity":
            mi[r["call"]][r["depth"]] = num(r["value"])

    fig = plt.figure(figsize=(7.4, 5.3))
    gs4 = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.62,
                           wspace=0.26)
    ax1 = fig.add_subplot(gs4[0, 0])
    ax2 = fig.add_subplot(gs4[0, 1])
    ax3 = fig.add_subplot(gs4[1, :])
    x = np.arange(4)


    dy = {"conserved": 0, "diagnostic": 1, "tolerant": 0, "indeterminate": -10}
    for call, col, mk in calls:
        y = [mi[call].get(d) for d in depths]
        if any(v is None for v in y):
            continue
        ax1.plot(x, y, color=col, marker=mk, zorder=3, label=call)
        ax1.annotate("%.2f" % y[-1], (x[-1], y[-1]), xytext=(6, dy[call]),
                     textcoords="offset points", color=col, fontsize=7.5, va="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels(nice)
    ax1.set_xlim(-0.25, 3.5)


    ax1.set_ylim(0, 1.52)
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.legend(loc="upper right", ncol=2, columnspacing=1.0)
    tidy(ax1, "mean identity to the human residue", "comparator set",
         "a  One position, four sequence sets")

    g3 = {r["depth"]: num(r["value"]) for r in dl if r["measure"] == "grade_shift_ge3_rate"}
    flip = {r["depth"]: num(r["value"]) for r in dl
            if r["measure"] == "grade_flip_variable_to_conserved"}
    sh = {r["depth"]: num(r["value"]) for r in dl if r["measure"] == "mean_abs_grade_shift"}
    dd = ["D1_paralogues", "D3_subfamily", "D4_family"]
    lab = ["3 paralogues", "16 subfamily", "79 family"]
    xx = np.arange(3)
    w = 0.36
    a = [100 * g3[d] for d in dd]
    b = [100 * flip[d] for d in dd]
    ax2.bar(xx - w / 2 - 0.012, a, w, color=S2, label="shifted $\\geq$3 grades")
    ax2.bar(xx + w / 2 + 0.012, b, w, color=S1, label="flipped variable to conserved")
    for xi, v in zip(xx - w / 2 - 0.012, a):
        ax2.annotate("%.1f%%" % v, (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5, color=INK)
    for xi, v in zip(xx + w / 2 + 0.012, b):
        ax2.annotate("%.1f%%" % v, (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=7.5, color=INK)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(["%s\nshift %.2f" % (l, sh[d]) for l, d in zip(lab, dd)],
                        fontsize=7.4)
    ax2.set_ylim(0, 54)
    ax2.legend(loc="upper left")
    tidy(ax2, "percent of positions", "comparator set",
         "b  The same change in ConSurf's units")


    fam = [r for r in rows("depth_ladder_families.csv")
           if r["depth"] == "D1_paralogues"]
    fam.sort(key=lambda r: num(r["mean_abs_shift"]))
    PRETTY = {"na_channels": "sodium channels", "peroxidase": "peroxidases",
              "fgfrs": "FGFRs", "jams": "JAMs", "ephs": "Eph receptors",
              "gjbs": "connexins"}
    labs = [PRETTY.get(r["family"], r["family"]) for r in fam]
    vals = [num(r["mean_abs_shift"]) for r in fam]
    cols = [S2 if r["family"] == "peroxidase" else S1 for r in fam]
    y = np.arange(len(fam))
    ax3.barh(y, vals, 0.62, color=cols)
    for yi, v, r in zip(y, vals, fam):
        ax3.annotate("%.2f" % v, (v, yi), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=7.0,
                     color=INK)
    med = float(np.median(vals))
    ax3.axvline(med, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax3.annotate("median %.2f" % med, (med, len(fam) - 0.35), xytext=(5, 0),
                 textcoords="offset points", fontsize=7.2, color=MUTED)
    ax3.set_yticks(y)
    ax3.set_yticklabels(labs, fontsize=7.4)
    ax3.set_xlim(0, 3.0)
    npos = sum(int(r["n_positions"]) for r in fam)
    tidy(ax3, None, "mean |grade shift| at paralogue depth, %s positions "
                    "over %d families" % (format(npos, ","), len(fam)),
         "c  The same measurement on every family, not one protein", grid="x")
    save(fig, "Figure3")


FAMILIES = [("peroxidase", "peroxidases", 17, 4),
            ("syndecans_reviewed", "syndecans\n(reviewed)", 18, 4),
            ("syndecans", "syndecans", 120, 4), ("nectins", "nectins", 120, 4),
            ("jams", "JAMs", 90, 3), ("fgfrs", "FGFRs", 120, 4),
            ("gjbs", "connexins", 150, 5), ("scns", "Na\nchannels", 200, 10),
            ("ephs", "Ephs", 280, 14)]


def family_rates():
    out = []
    for key, label, nseq, nsub in FAMILIES:
        if not os.path.exists(os.path.join(DATA, "tree_null_%s.csv" % key)):
            continue


        rs = [r for r in rows("tree_null_%s.csv" % key)
              if r["call_tree"] not in ("", "conserved", None)]
        if not rs:
            continue
        n = len(rs)
        out.append(dict(key=key, label=label, nseq=nseq, nsub=nsub, n=n,
                        perm=sum(1 for r in rs if r["call_perm"] == "diagnostic") / n,
                        tree=sum(1 for r in rs if r["call_tree"] == "diagnostic") / n))
    return out


TREE_CACHE = "figure5_peroxidase_tree.csv"
SUBS5 = {"LPO": ("Lactoperoxidase",), "TPO": ("Thyroid peroxidase",),
         "EPX": ("Eosinophil peroxidase",), "MPO": ("Myeloperoxidase",)}


def build_tree_cache():

    spec = importlib.util.spec_from_file_location(
        "treenull", os.path.join(HERE, "07_tree_aware_null.py"))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    import partition as P
    src = P.local_data("an_peroxidase_PF03098.fasta")
    if not os.path.exists(src):
        raise SystemExit("fatal: peroxidase FASTA not found at %s" % src)
    fam = P.read_fasta(src)
    labels, _ = P.assign_labels(fam, SUBS5)
    msa = P.align({k: v for k, v in fam.items() if k in labels})
    keys = sorted(msa)
    quiet = lambda *a, **k: None
    D = tm.p_distance_matrix(msa, keys, quiet)
    root, _ = tm.neighbour_joining(D, keys, quiet)

    segs, ypos = [], [0.0]

    def walk(node, x):
        name, kids = node
        if not kids:
            y = ypos[0]
            ypos[0] += 1.0
            segs.append((x, y, x + 0.05, y, "tip", labels.get(name, "")))
            return y
        ys = []
        for child, length in kids:
            cy = walk(child, x + max(length, 0.006))
            segs.append((x, cy, x + max(length, 0.006), cy, "branch", ""))
            ys.append(cy)
        segs.append((x, min(ys), x, max(ys), "branch", ""))
        return float(np.mean(ys))

    walk(root, 0.0)
    out = os.path.join(DATA, TREE_CACHE)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x0", "y0", "x1", "y1", "kind", "subfamily"])
        for r in segs:
            w.writerow(["%.6f" % r[0], "%.6f" % r[1], "%.6f" % r[2],
                        "%.6f" % r[3], r[4], r[5]])
    print("wrote %s (%d segments, %d tips)"
          % (TREE_CACHE, len(segs), sum(1 for r in segs if r[4] == "tip")))


def draw_tree(axt):
    path = os.path.join(DATA, TREE_CACHE)
    if not os.path.exists(path):
        raise SystemExit(
            "fatal: %s is missing. Figure 4a is the real neighbour-joining tree "
            "and is not optional. Rebuild the cache with\n"
            "    python3 build_figures.py --rebuild-tree\n"
            "which needs pyfamsa; drawing from the cache does not." % TREE_CACHE)
    segs = rows(TREE_CACHE)
    colmap = {"LPO": S1, "EPX": S2, "TPO": S3, "MPO": S4}
    ntip = 0
    for r in segs:
        x0, y0, x1, y1 = (num(r[k]) for k in ("x0", "y0", "x1", "y1"))
        if r["kind"] == "tip":
            axt.plot([x0, x1], [y0, y1], lw=2.2, solid_capstyle="round",
                     color=colmap.get(r["subfamily"], MUTED))
            ntip += 1
        else:
            axt.plot([x0, x1], [y0, y1], color=MUTED, lw=1.0)
    if ntip != 17:
        raise SystemExit("fatal: tree cache holds %d tips, expected the 17 "
                         "mammalian-type peroxidases" % ntip)
    axt.set_yticks([])
    axt.set_xticks([])
    for sp in ("top", "right", "left", "bottom"):
        axt.spines[sp].set_visible(False)
    axt.legend(handles=[Line2D([0], [0], color=c, lw=2.6, label=k)
                        for k, c in colmap.items()],
               loc="lower right", ncol=2, fontsize=7.5)
    axt.set_title("a  The tree the null is simulated on", loc="left",
                  color=INK, pad=7)


def figure5():
    fr = family_rates()


    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.2, 1],
                          hspace=0.62, wspace=0.28)

    axt = fig.add_subplot(gs[0, 0])
    draw_tree(axt)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(fr))
    w = 0.37
    a = [100 * f["perm"] for f in fr]
    b = [100 * f["tree"] for f in fr]
    ax1.bar(x - w / 2 - 0.012, a, w, color=S2, label="label-permutation null")
    ax1.bar(x + w / 2 + 0.012, b, w, color=S1, label="tree-aware null")
    for k, (xi, v) in enumerate(zip(x - w / 2 - 0.012, a)):
        ax1.annotate("%.1f" % v, (xi, v), xytext=(0, 3 if k % 2 == 0 else 13),
                     textcoords="offset points",
                     ha="center", fontsize=6.8, color=INK)
    for xi, v in zip(x + w / 2 + 0.012, b):
        ax1.annotate("%.1f" % v, (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=6.8, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f["label"].replace("\n", " ").replace("(reviewed)", "rev.")
                         for f in fr], fontsize=7.0, rotation=28, ha="right")
    ax1.set_ylim(0, 132)
    ax1.legend(loc="upper center", fontsize=7.2, ncol=1, handlelength=1.4)
    tidy(ax1, "variable columns diagnostic (%)", None,
         "b  The permutation null is uninformative")

    ax2 = fig.add_subplot(gs[1, 0])
    pair = [f for f in fr if f["key"] in ("syndecans_reviewed", "syndecans")]
    if len(pair) == 2:
        xs = np.arange(2)
        vals = [100 * p["tree"] for p in pair]
        ax2.bar(xs, vals, 0.46, color=[S4, S1])
        for xi, v in zip(xs, vals):
            ax2.annotate("%.1f%%" % v, (xi, v), xytext=(0, 4),
                         textcoords="offset points", ha="center", fontsize=9.5,
                         color=INK, weight="bold")
        ax2.set_xticks(xs)
        ax2.set_xticklabels(["reviewed only\n18 seq, 3 to 6 per gene",
                             "with unreviewed\n120 seq, 30 per gene"], fontsize=7.8)
        ax2.set_ylim(0, 37)
        tidy(ax2, "diagnostic under the tree null (%)", None,
             "c  Same genes, two sampling depths")

    ax3 = fig.add_subplot(gs[1, 1])
    deep = [f for f in fr if f["nseq"] >= 90]
    xs = [f["nsub"] for f in deep]
    ys = [100 * f["tree"] for f in deep]
    ax3.scatter(xs, ys, s=62, color=S1, zorder=3, edgecolor=WHITE, linewidth=0.9)


    rho = spearman(xs, ys)


    want = {}
    want[max(range(len(deep)), key=lambda i: ys[i])] = (-6, 10, "right")
    want[min(range(len(deep)), key=lambda i: ys[i])] = (10, -4, "left")
    for i, f in enumerate(deep):
        if f["label"].replace("\n", " ") == "FGFRs":
            want[i] = (10, 2, "left")
    for i, (dx, dy, ha) in want.items():
        ax3.annotate("%s %.1f%%" % (deep[i]["label"].replace("\n", " "), ys[i]),
                     (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                     ha=ha, fontsize=7.2, color=INK)
    ax3.annotate("Spearman %.2f over %d families" % (rho, len(deep)),
                 (0.97, 0.05), xycoords="axes fraction", ha="right",
                 fontsize=7.4, color=INK2)
    ax3.set_xlim(1.6, 17.0)
    ax3.set_ylim(0, 92)
    ax3.set_xticks([3, 4, 5, 10, 14])
    tidy(ax3, "diagnostic under the tree null (%)", "subfamilies in the family",
         "d  Subfamily count, loosely")
    save(fig, "Figure5")


def _survival_panels(ax1, ax2):
    gsm = rows("gap_survival_summary.csv")
    variants = ["A_as_published", "B_gap_filter", "C_gap_corrected", "D_both"]
    nice = ["as\npublished", "gap\nfilter", "gap-free\nconservation", "both"]
    cohorts = [("data-human", "human", S1), ("data-alphahelical", "alpha-helical", S2)]

    def grab(measure, cohort, variant):
        for r in gsm:
            if (r["measure"] == measure and r["cohort"] == cohort
                    and r["variant"] == variant):
                return num(r["value"])
        return None

    x = np.arange(4)
    for i, (ck, clab, col) in enumerate(cohorts):
        y = [grab("auc_rcs10", ck, v) for v in variants]
        ax1.plot(x, y, color=col, marker="o", zorder=3, label=clab)
        for xi in (0, len(x) - 1):
            ax1.annotate("%.3f" % y[xi], (x[xi], y[xi]),
                         xytext=(0, 8 if i == 0 else -15),
                         textcoords="offset points", ha="center", fontsize=7.5,
                         color=col)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nice, fontsize=7.8)
    ax1.set_xlim(-0.25, 3.25)
    ax1.set_ylim(0.73, 0.99)
    ax1.legend(loc="lower right")
    tidy(ax1, "AUC of the rigid conserved score", "pipeline variant",
         "a  Gap handling raises the AUC")


    gis = rows("gap_inflation_summary.csv")

    def gis_val(measure, cohort):
        for r in gis:
            if r["measure"] == measure and r["cohort"] == cohort:
                return num(r["value"])
        raise SystemExit("fatal: %s / %s missing from gap_inflation_summary.csv"
                         % (measure, cohort))

    names = ["DCS\ntop decile", "DCS\ntop 1%", "RCS\ntop decile", "RCS\ntop 1%"]
    keys = ["gappy_share_Dynamic_Conserved_Score_0_10_topdecile",
            "gappy_share_Dynamic_Conserved_Score_0_10_top1%",
            "gappy_share_Rigid_Conserved_Score_0_10_topdecile",
            "gappy_share_Rigid_Conserved_Score_0_10_top1%"]
    base, vals = {}, {}
    for ck, clab, col in cohorts:
        base[ck] = gis_val("base_rate_gappy", ck)
        vals[ck] = [gis_val(k, ck) for k in keys]
    xx = np.arange(len(names))
    w = 0.37
    handles = []
    for i, (ck, clab, col) in enumerate(cohorts):
        off = (-w / 2 - 0.012) if i == 0 else (w / 2 + 0.012)
        b = ax2.bar(xx + off, [100 * v for v in vals[ck]], w, color=col,
                    label=clab)
        handles.append(b)
        for xi, v in zip(xx + off, vals[ck]):
            ax2.annotate("%.0f" % (100 * v), (xi, 100 * v), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=7.5,
                         color=INK)
        ax2.axhline(100 * base[ck], color=col, lw=1.1, ls=(0, (4, 3)), alpha=0.9)
        handles.append(Line2D([0], [0], color=col, lw=1.1, ls=(0, (4, 3)),
                              label="baseline %.1f%%" % (100 * base[ck])))
    ax2.set_xticks(xx)
    ax2.set_xticklabels(names, fontsize=7.8)
    ax2.set_ylim(0, 112)
    ax2.legend(handles=handles, loc="upper left", fontsize=7.0,
               ncol=2, columnspacing=1.0, handlelength=1.6, borderpad=0.4)
    tidy(ax2, "residues on >50%-gap columns (%)", None,
         "b  Top-scoring tails are gap-rich")


def figure6():
    byfam = rows("sister_state_by_family.csv")
    summ = {r["arm"]: r for r in rows("sister_state_summary.csv")}
    arm = "all mapped positions"
    fams = [r for r in byfam if r["arm"] == arm]
    fams = [r for r in fams
            if int(r["sister_benign"]) + int(r["sister_pathogenic"]) > 0]
    fams.sort(key=lambda r: -(int(r["other_benign"])
              / max(int(r["other_benign"]) + int(r["other_pathogenic"]), 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(fams))
    h = 0.36
    sis, oth = [], []
    for r in fams:
        a, b = int(r["sister_benign"]), int(r["sister_pathogenic"])
        c, d = int(r["other_benign"]), int(r["other_pathogenic"])
        sis.append(100 * a / (a + b))
        oth.append(100 * c / (c + d) if c + d else 0.0)
    ax1.barh(y + h / 2 + 0.01, sis, h, color=S1, label="to a sister state")
    ax1.barh(y - h / 2 - 0.01, oth, h, color=GREY, label="any other substitution")
    for yy, v, n in zip(y + h / 2 + 0.01, sis,
                        [int(r["sister_benign"]) + int(r["sister_pathogenic"])
                         for r in fams]):
        ax1.annotate("%.0f%%  (n=%d)" % (v, n), (v, yy), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=6.8,
                     color=INK)
    for yy, v, n in zip(y - h / 2 - 0.01, oth,
                        [int(r["other_benign"]) + int(r["other_pathogenic"])
                         for r in fams]):
        ax1.annotate("%.0f%%  (n=%d)" % (v, n), (v, yy), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=6.8,
                     color=MUTED)
    ax1.set_yticks(y)
    ax1.set_yticklabels([r["family"] for r in fams], fontsize=7.8)
    ax1.set_xlim(0, 132)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.invert_yaxis()
    ax1.legend(loc="lower right", fontsize=7.2, framealpha=0.95)
    tidy(ax1, None, "variants classified benign (%)",
         "a  Benign share, sister states against the rest", grid="x")


    labels, ors, los, his, ps, cols = [], [], [], [], [], []
    labels.append("pooled,\nunstratified")
    ors.append(num(summ["all_mapped"]["or_crude"]))
    los.append(num(summ["all_mapped"]["crude_ci_low"]))
    his.append(num(summ["all_mapped"]["crude_ci_high"]))
    ps.append(None)
    cols.append(GREY)
    for key, lab, col in (("all_mapped", "stratified,\nall positions", S1),
                          ("tree_diagnostic",
                           "stratified,\ntree-diagnostic", S3)):
        labels.append(lab)
        ors.append(num(summ[key]["or_mh"]))
        los.append(num(summ[key]["ci_low"]))
        his.append(num(summ[key]["ci_high"]))
        ps.append(num(summ[key]["p_mh"]))
        cols.append(col)
    x = np.arange(len(ors))
    err = np.array([[v - lo for v, lo in zip(ors, los)],
                    [hi - v for v, hi in zip(ors, his)]])
    if (err < 0).any():
        raise SystemExit("fatal: a deposited interval does not bracket its own "
                         "estimate, so Figure 6b would draw a negative whisker")


    BARW = 0.46
    ax2.bar(x, ors, BARW, color=cols, yerr=err, capsize=3.5, ecolor=INK,
            error_kw={"lw": 1.0, "zorder": 3})
    for xi, v, lo, hi, pv in zip(x, ors, los, his, ps):
        txt = "%.2f\n%.2f to %.2f" % (v, lo, hi)
        if pv is not None:
            txt += "\np = %.4f" % pv
        ax2.annotate(txt, (xi, hi), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=6.6, color=INK)
    ax2.axhline(1.0, color=S2, lw=1.1, ls=(0, (4, 3)))


    noeff = ax2.annotate("no effect", (1.5, 1.0), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=6.5,
                         color=S2, annotation_clip=False,
                         bbox=dict(facecolor="white", edgecolor="none",
                                   pad=0.8, alpha=0.9))
    x0, x1 = ax2.get_xlim()
    if not x0 < 1.5 < x1:
        raise SystemExit("fatal: the no-effect label sits outside Figure 6b's "
                         "axes, where it would be drawn clipped or not at all")
    fig.canvas.draw()
    lab = noeff.get_window_extent(fig.canvas.get_renderer())
    gap = (ax2.transData.transform((1.0 + BARW / 2, 0))[0],
           ax2.transData.transform((2.0 - BARW / 2, 0))[0])
    if lab.x0 < gap[0] or lab.x1 > gap[1]:
        raise SystemExit("fatal: the no-effect label is wider than the gap "
                         "between the bars it sits between, so it covers them")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.4)
    tidy(ax2, "odds ratio, benign", None,
         "b  Most of the naive effect is the family")
    ax2.set_ylim(0, max(his) * 1.38)
    fig.tight_layout()
    save(fig, "Figure6")


def figure2():
    fig = plt.figure(figsize=(7.4, 3.1))
    gsg = fig.add_gridspec(1, 2, wspace=0.30, width_ratios=[1.3, 1])
    _gap_panels(fig.add_subplot(gsg[0, 0]), fig.add_subplot(gsg[0, 1]))
    save(fig, "Figure2")


def figureS4():
    fig = plt.figure(figsize=(7.4, 3.1))
    gsg = fig.add_gridspec(1, 2, wspace=0.32, width_ratios=[1.15, 1])
    _survival_panels(fig.add_subplot(gsg[0, 0]), fig.add_subplot(gsg[0, 1]))
    save(fig, "FigureS4")


def figureS1():

    fr = family_rates()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    labs, fail, edge = [], [], []
    for f in fr:
        rs = rows("tree_null_%s.csv" % f["key"])
        tested = [r for r in rs if r["call_tree"] not in ("", "conserved", None)]
        att = [r for r in rs if r["parsimony_obs"] not in ("", None)]
        if not att:
            continue
        labs.append(f["label"].replace("\n", " "))
        fail.append(100 * sum(1 for r in att if r["calibrated"] != "yes") / len(att))
        edge.append(100 * sum(1 for r in tested
                              if r["rate_at_grid_edge"] == "yes") / max(len(tested), 1))
    x = np.arange(len(labs))
    w = 0.38
    ax1.bar(x - w / 2 - 0.01, fail, w, color=S2, label="calibration failed")
    ax1.bar(x + w / 2 + 0.01, edge, w, color=S4, label="rate at a grid edge")
    for xi, v in zip(x - w / 2 - 0.01, fail):
        ax1.annotate("%.1f" % v, (xi, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=6.8, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labs, fontsize=7.0, rotation=28, ha="right")
    ax1.set_ylim(0, 62)
    ax1.legend(loc="upper left", fontsize=7.0)
    tidy(ax1, "per cent of columns", None, "a  Calibration diagnostics by family")

    rs = [r for r in rows("tree_null_peroxidase.csv")
          if r["parsimony_obs"] not in ("", None) and r["calibrated"] == "yes"]
    po = np.array([num(r["parsimony_obs"]) for r in rs], dtype=float)
    pn = np.array([num(r["parsimony_null"]) for r in rs], dtype=float)
    ax2.scatter(po, pn, s=12, color=S1, alpha=0.45, edgecolor="none")
    hi = float(max(po.max(), pn.max())) * 1.05
    ax2.plot([0, hi], [0, hi], color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax2.annotate("y = x", (hi * 0.80, hi * 0.86), fontsize=7.0, color=MUTED)
    dev = float(np.max(np.abs(po - pn)))
    ax2.annotate("worst |observed − simulated| = %.2f substitutions" % dev,
                 (0.96, 0.06), xycoords="axes fraction", ha="right",
                 fontsize=7.0, color=INK2)
    ax2.set_xlim(0, hi)
    ax2.set_ylim(0, hi)
    tidy(ax2, "mean simulated parsimony", "observed Fitch parsimony",
         "b  What the search matches on")
    fig.tight_layout()
    save(fig, "FigureS1")


def figureS2():

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    rs = [r for r in rows("tree_null_peroxidase.csv")
          if r["sef"] not in ("", None) and r["calibrated"] == "yes"]
    sef = np.array([num(r["sef"]) for r in rs], dtype=float)
    tree = np.array([num(r["sef_null_tree"]) for r in rs], dtype=float)


    adjp = np.array([num(r["sef_adj_perm"]) for r in rs], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        permE = np.where(adjp < 1.0, (sef - adjp) / (1.0 - adjp), np.nan)
    bins = np.linspace(0, 1, 26)
    ax1.hist(sef, bins=bins, color=S1, alpha=0.85, label="observed")
    ax1.hist(tree, bins=bins, color=S3, alpha=0.70, label="tree null expectation")
    ax1.hist(permE[~np.isnan(permE)], bins=bins, color=S2, alpha=0.55,
             label="permutation null expectation")
    ax1.legend(loc="upper left", fontsize=7.0)
    tidy(ax1, "columns", "subfamily-explained fraction",
         "a  Peroxidases, where each null sits")

    labs, gaps = [], []
    for f in family_rates():
        rs = [r for r in rows("tree_null_%s.csv" % f["key"])
              if r["sef"] not in ("", None) and r["calibrated"] == "yes"]
        if not rs:
            continue
        adjp = np.array([num(r["sef_adj_perm"]) for r in rs], dtype=float)
        adjt = np.array([num(r["sef_adj_tree"]) for r in rs], dtype=float)
        labs.append(f["label"].replace("\n", " "))
        gaps.append((float(adjp.mean()), float(adjt.mean())))
    y = np.arange(len(labs))
    for yi, (a, b) in zip(y, gaps):
        ax2.plot([b, a], [yi, yi], color=MUTED, lw=1.0, zorder=1)
    ax2.scatter([g[0] for g in gaps], y, s=34, color=S2, zorder=3,
                label="permutation")
    ax2.scatter([g[1] for g in gaps], y, s=34, color=S1, zorder=3, label="tree")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labs, fontsize=7.2)
    ax2.set_xlim(0, 1.0)
    ax2.invert_yaxis()
    ax2.legend(loc="center left", fontsize=7.0, framealpha=0.95)
    tidy(ax2, None, "mean chance-corrected value", "b  The correction each null applies",
         grid="x")
    fig.tight_layout()
    save(fig, "FigureS2")


def figureS3():

    import gzip as _gz
    per = defaultdict(lambda: [0.0, 0, None])
    with _gz.open(os.path.join(DATA, "gap_inflation_positions.csv.gz"), "rt") as fh:
        for r in csv.DictReader(fh):
            k = (r["cohort"], r["protein_id"])
            per[k][0] += float(r["gap_frac"])
            per[k][1] += 1
            per[k][2] = float(r["bitscore"])
    pts = [(v[2], v[0] / v[1], k[0]) for k, v in per.items() if v[1]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    gf = np.array([p[1] for p in pts], dtype=float)
    ax1.hist(gf, bins=24, color=S1)
    ax1.axvline(float(gf.mean()), color=S2, lw=1.2, ls=(0, (4, 3)))
    ax1.annotate("mean %.3f" % gf.mean(), (0.97, 0.93), xycoords="axes fraction",
                 ha="right", fontsize=7.2, color=S2)
    tidy(ax1, "proteins", "mean gap fraction of the protein's columns",
         "a  Gappiness varies widely between proteins")

    jitter = (np.arange(len(pts)) % 7 - 3) * 0.004
    for ck, lab, col in (("data-human", "human", S1),
                         ("data-alphahelical", "alpha-helical", S2)):
        sel = [(p[0], p[1]) for p in pts if p[2] == ck]
        j = (np.arange(len(sel)) % 7 - 3) * 0.004
        ax2.scatter([s[0] + jj for s, jj in zip(sel, j)], [s[1] for s in sel],
                    s=22, color=col, alpha=0.65, edgecolor="none", label=lab)
    ax2.annotate("Spearman %+.3f over %d proteins"
                 % (spearman([p[0] for p in pts], [p[1] for p in pts]), len(pts)),
                 (0.96, 0.93),
                 xycoords="axes fraction", ha="right", fontsize=7.2, color=INK2)
    ax2.set_xlim(0.0, 0.8)
    ax2.legend(loc="upper left", fontsize=7.0)
    tidy(ax2, "mean gap fraction", "jackhmmer bitscore threshold",
         "b  Depth does not predict gappiness")
    fig.tight_layout()
    save(fig, "FigureS3")


if __name__ == "__main__":
    if "--rebuild-tree" in sys.argv:
        build_tree_cache()
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figureS1()
    figureS2()
    figureS3()
    figureS4()
    print("\nall figures written to", FIGS)
