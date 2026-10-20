#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

import partition as P


PEROX_FUNCTIONAL = (225, 226, 227, 301, 303, 305, 307, 372, 375, 468)
PEROX_DESIGNED = (144, 648, 205, 313)
PEROX_HELD_BACK = (479, 169)
PEROX_NATURAL = (414, 514, 700)

N_SIM = 500
N_CAL = 80
RATE_GRID = np.exp(np.linspace(math.log(0.02), math.log(60.0), 16))
PARS_TOL = 0.50

SEED = 20261007


def p_distance_matrix(msa, keys, log):

    n = len(keys)
    arr = [np.frombuffer(msa[k].encode(), dtype="S1") for k in keys]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = arr[i], arr[j]
            both = (a != b"-") & (b != b"-")
            m = int(both.sum())
            if m == 0:
                raise P.Fatal("sequences %s and %s share no ungapped column"
                              % (keys[i][:30], keys[j][:30]))
            diff = int((a[both] != b[both]).sum())
            p = diff / m
            p = min(p, 0.94)
            d = -math.log(1.0 - p)
            D[i, j] = D[j, i] = d
    if not np.isfinite(D).all():
        raise P.Fatal("distance matrix carries a non-finite entry")
    log("  distance matrix %dx%d, p-distance range %.3f to %.3f"
        % (n, n, D[np.triu_indices(n, 1)].min(), D[np.triu_indices(n, 1)].max()))
    return D


def neighbour_joining(D, keys, log):

    n0 = len(keys)
    nodes = [(k, []) for k in keys]
    D = D.copy().astype(float)
    active = list(range(len(nodes)))
    neg = 0
    while len(active) > 2:
        m = len(active)
        r = {i: sum(D[i, j] for j in active if j != i) for i in active}
        best, bi, bj = None, None, None
        for ai in range(m):
            for aj in range(ai + 1, m):
                i, j = active[ai], active[aj]
                q = (m - 2) * D[i, j] - r[i] - r[j]
                if best is None or q < best:
                    best, bi, bj = q, i, j
        li = 0.5 * D[bi, bj] + (r[bi] - r[bj]) / (2.0 * (m - 2))
        lj = D[bi, bj] - li
        if li < 0:
            neg += 1
            li = 0.0
        if lj < 0:
            neg += 1
            lj = 0.0
        new = (None, [(nodes[bi], li), (nodes[bj], lj)])
        nodes.append(new)
        k = len(nodes) - 1
        newrow = np.zeros(len(nodes))
        D = np.pad(D, ((0, 1), (0, 1)), mode="constant")
        for x in active:
            if x in (bi, bj):
                continue
            D[k, x] = D[x, k] = 0.5 * (D[bi, x] + D[bj, x] - D[bi, bj])
        active = [x for x in active if x not in (bi, bj)] + [k]
    i, j = active
    ln = max(0.0, D[i, j])
    root = (None, [(nodes[i], ln / 2.0), (nodes[j], ln / 2.0)])
    tips = []

    def walk(node):
        name, kids = node
        if not kids:
            tips.append(name)
        for c, _ in kids:
            walk(c)
    walk(root)
    if sorted(tips) != sorted(keys):
        raise P.Fatal("the tree has %d tips against %d sequences"
                      % (len(tips), n0))
    log("  neighbour-joining tree built, %d tips, %d negative branch lengths "
        "clamped to zero" % (len(tips), neg))
    return root, neg


def simulate(root, tip_index, pi, rate, n_sim, rng):

    k = len(pi)
    beta = 1.0 / max(1e-12, 1.0 - float((pi ** 2).sum()))
    out = np.empty((n_sim, len(tip_index)), dtype=np.int16)
    start = rng.choice(k, size=n_sim, p=pi)

    def down(node, state):
        name, kids = node
        if not kids:
            out[:, tip_index[name]] = state
            return
        for child, length in kids:
            t = length * rate
            keep = math.exp(-beta * t) if t > 0 else 1.0
            redraw = rng.random(n_sim) > keep
            nxt = state.copy()
            if redraw.any():
                nxt[redraw] = rng.choice(k, size=int(redraw.sum()), p=pi)
            down(child, nxt)
    down(root, start)
    return out


def post_order(root, tip_index):

    plan, slots = [], {}

    def visit(node):
        name, kids = node
        if not kids:
            slot = len(plan)
            plan.append(("tip", tip_index[name], None))
            return slot
        child_slots = [visit(c) for c, _ in kids]
        slot = len(plan)
        plan.append(("node", None, child_slots))
        return slot
    root_slot = visit(root)
    return plan, root_slot


def fitch_vec(plan, root_slot, states, n_slots=None):

    n_rep = states.shape[0]
    scratch = [None] * len(plan)
    changes = np.zeros(n_rep, dtype=np.int32)
    for slot, (kind, tip, kids) in enumerate(plan):
        if kind == "tip":
            scratch[slot] = (np.int64(1) << states[:, tip].astype(np.int64))
        else:
            acc = scratch[kids[0]]
            for cs in kids[1:]:
                m = scratch[cs]
                inter = acc & m
                empty = inter == 0
                acc = np.where(empty, acc | m, inter)
                changes += empty
            scratch[slot] = acc
    return changes


def sef_rows(states, lab_idx, n_lab):

    n_sim, n_tip = states.shape
    k = int(states.max()) + 1 if states.size else 1
    off = (k * np.arange(n_sim, dtype=np.int64))[:, None]

    def counts(sub):
        if sub.shape[1] == 0:
            return np.zeros((n_sim, k))
        flat = (sub.astype(np.int64) + off).ravel()
        return np.bincount(flat, minlength=n_sim * k).reshape(n_sim, k)

    def ent(c, n):
        with np.errstate(divide="ignore", invalid="ignore"):
            p = c / n[:, None]
            t = np.where(p > 0, p * np.log2(p), 0.0)
        return -t.sum(axis=1)

    tot = counts(states)
    n_all = np.full(n_sim, float(n_tip))
    h = ent(tot, n_all)
    hw = np.zeros(n_sim)
    for l in range(n_lab):
        sel = states[:, lab_idx == l]
        if sel.shape[1] == 0:
            continue
        c = counts(sel)
        nl = float(sel.shape[1])
        hw += (nl / n_tip) * ent(c, np.full(n_sim, nl))
    sefs = np.where(h > 0, np.maximum(0.0, h - hw) / np.where(h > 0, h, 1.0), 0.0)
    return sefs, h


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta",
                    default=P.local_data("an_peroxidase_PF03098.fasta"))
    ap.add_argument("--name", default="peroxidase",
                    help="family name, used in output filenames and to decide "
                         "whether the peroxidase annotation applies")
    ap.add_argument("--ref", default="PERL_HUMAN")
    ap.add_argument("--own", default="LPO")
    ap.add_argument("--subfamilies", default=None,
                    help='"LAB=pattern;LAB=pattern"; defaults to the four '
                         'mammalian-type peroxidases')
    ap.add_argument("--outdir", default=os.path.join("..", "data"))
    ap.add_argument("--n-sim", type=int, default=N_SIM)
    ap.add_argument("--n-cal", type=int, default=N_CAL)
    a = ap.parse_args()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("07_tree_aware_null.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("Is a column more clade-partitioned than a neutral character on this tree?")
    log("")
    log("model: F81 with the column's own composition as equilibrium, equal")
    log("exchangeability, branch lengths rate-matched per column to the observed")
    log("Fitch parsimony score. n_sim %d, calibration %d per grid point, grid "
        "%.2f to %.1f," % (a.n_sim, N_CAL, RATE_GRID[0], RATE_GRID[-1]))
    log("parsimony tolerance max(%.2f, 5%% of the score), seed %d" % (PARS_TOL, SEED))

    if a.subfamilies:
        subs = {}
        for part in a.subfamilies.split(";"):
            if not part.strip():
                continue
            lab, pat = part.split("=", 1)
            subs.setdefault(lab.strip(), []).append(pat)
        SUBFAMILIES = {k: tuple(v) for k, v in subs.items()}
    else:
        SUBFAMILIES = {"LPO": ("Lactoperoxidase",),
                       "TPO": ("Thyroid peroxidase",),
                       "EPX": ("Eosinophil peroxidase",),
                       "MPO": ("Myeloperoxidase",)}
    OWN = a.own
    is_perox = a.name == "peroxidase"
    FUNCTIONAL = PEROX_FUNCTIONAL if is_perox else ()
    DESIGNED = PEROX_DESIGNED if is_perox else ()
    HELD_BACK = PEROX_HELD_BACK if is_perox else ()
    NATURAL = PEROX_NATURAL if is_perox else ()

    fam = P.read_fasta(a.fasta)
    ref = next((k for k in fam if a.ref in k), None)
    if ref is None:
        raise P.Fatal("no header containing %r in %s" % (a.ref, a.fasta))
    ref_seq = fam[ref]
    labels, _ = P.assign_labels(fam, SUBFAMILIES)
    if labels.get(ref) != OWN:
        raise P.Fatal("the reference labelled %r, not %r"
                      % (labels.get(ref), OWN))
    sub = {k: v for k, v in fam.items() if k in labels}
    msa = P.align(sub)
    pos_to_col = P.map_positions(msa, ref, ref_seq)
    log("\naligned %d sequences, width %d, reference row reconstructs its input"
        % (len(sub), len(next(iter(msa.values())))))

    keys = sorted(msa)
    D = p_distance_matrix(msa, keys, log)
    root, _neg = neighbour_joining(D, keys, log)
    tip_index = {k: i for i, k in enumerate(keys)}
    plan, root_slot = post_order(root, tip_index)
    log("  post-order plan with %d slots" % len(plan))


    comp_keys = [k for k in keys if k != ref]
    lab_names = sorted({labels[k] for k in keys})
    lab_of = {k: lab_names.index(labels[k]) for k in keys}

    rng = np.random.default_rng(SEED)
    perm_rng = np.random.default_rng(SEED)

    rows, failed_cal, edge_hits = [], 0, [0]
    for pos in sorted(pos_to_col):
        col = pos_to_col[pos]
        wt = ref_seq[pos - 1]
        res, labs, gap_frac, dropped = P.column_set(msa, col, ref, labels)
        perm = P.column_partition(res, labs, perm_rng, n_perm=1000)
        call_perm = P.classify(perm)
        row = dict(position=pos, wt=wt, gap_frac=round(gap_frac, 4),
                   n=perm["n"], h_total=round(perm["h_total"], 4),
                   sef=None if perm["sef"] is None else round(perm["sef"], 4),
                   sef_adj_perm=None if perm["sef_adj"] is None
                   else round(perm["sef_adj"], 4),
                   p_perm=perm["p_perm"], call_perm=call_perm,
                   set=("functional" if pos in FUNCTIONAL else
                        "designed" if pos in DESIGNED else
                        "held_back" if pos in HELD_BACK else
                        "natural_variant" if pos in NATURAL else ""))


        if (perm["sef"] is None or perm["h_total"] < P.H_MIN
                or gap_frac > P.MAX_GAP_FRAC):
            row.update(rate=None, sef_null_tree=None, p_tree=None,
                       sef_adj_tree=None, call_tree="conserved"
                       if perm["sef"] is None or perm["h_total"] < P.H_MIN
                       else "", calibrated="", rate_at_grid_edge="",
                       parsimony_obs=None, parsimony_null=None)
            rows.append(row)
            continue


        present = sorted(set(res))
        idx = {r: i for i, r in enumerate(present)}
        cnt = Counter(res)
        pi = np.array([cnt[r] for r in present], dtype=float)
        pi /= pi.sum()

        sim_tips = [k for k in keys]
        lab_idx_full = np.array([lab_of[k] for k in sim_tips])
        keep = np.array([k != ref for k in sim_tips])
        lab_idx = lab_idx_full[keep]


        obs_states = np.array([idx.get(msa[k][col], -1) for k in sim_tips])
        if (obs_states < 0).any():

            extra = sorted({msa[k][col] for k in sim_tips} - set(idx))
            for e in extra:
                idx[e] = len(idx)
            obs_states = np.array([idx[msa[k][col]] for k in sim_tips])
            pi = np.concatenate([pi, np.zeros(len(idx) - len(pi))])
            pi = pi + 1e-9
            pi /= pi.sum()
        target_p = int(fitch_vec(plan, root_slot,
                                 obs_states.reshape(1, -1))[0])


        def mean_pars(rate, n):
            st_ = simulate(root, tip_index, pi, float(rate), n, rng)
            return float(fitch_vec(plan, root_slot, st_).mean())

        lo = hi = None
        lo_v = hi_v = None
        for r in RATE_GRID:
            v = mean_pars(r, a.n_cal)
            if v <= target_p:
                lo, lo_v = float(r), v
            if v >= target_p and hi is None:
                hi, hi_v = float(r), v
                break
        if lo is None:
            best_rate, best_mean = float(RATE_GRID[0]), mean_pars(RATE_GRID[0], a.n_cal)
        elif hi is None:
            best_rate, best_mean = float(RATE_GRID[-1]), lo_v
        else:
            for _ in range(8):
                mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                v = mean_pars(mid, 2 * a.n_cal)
                if v < target_p:
                    lo, lo_v = mid, v
                else:
                    hi, hi_v = mid, v
            best_rate = math.exp(0.5 * (math.log(lo) + math.log(hi)))
            best_mean = mean_pars(best_rate, 4 * a.n_cal)
        best_gap = abs(best_mean - target_p)
        tol = max(PARS_TOL, 0.05 * target_p)
        calibrated = best_gap <= tol
        if not calibrated:
            failed_cal += 1
        at_edge = (best_rate <= RATE_GRID[0] * 1.001
                   or best_rate >= RATE_GRID[-1] * 0.999)
        if at_edge:
            edge_hits[0] += 1

        st = simulate(root, tip_index, pi, best_rate, a.n_sim, rng)
        sefs, hs = sef_rows(st[:, keep], lab_idx, len(lab_names))
        null_mean = float(sefs.mean())
        obs = perm["sef"]
        p_tree = (1.0 + float((sefs >= obs - 1e-12).sum())) / (a.n_sim + 1.0)
        den = 1.0 - null_mean
        adj = max(0.0, (obs - null_mean) / den) if den > 1e-12 else 0.0
        if adj >= P.SEF_HI and p_tree <= P.P_MAX:
            call_tree = "diagnostic"
        elif adj < P.SEF_LO:
            call_tree = "tolerant"
        else:
            call_tree = "indeterminate"
        row.update(rate=round(best_rate, 4), sef_null_tree=round(null_mean, 4),
                   p_tree=round(p_tree, 4), sef_adj_tree=round(adj, 4),
                   call_tree=call_tree,
                   calibrated="yes" if calibrated else "no",
                   rate_at_grid_edge="yes" if at_edge else "no",
                   parsimony_obs=target_p,
                   parsimony_null=round(best_mean, 3) if best_mean else None,
                   null_h_mean=round(float(hs.mean()), 4))
        rows.append(row)

    usable = [r for r in rows if r["call_tree"] not in ("", None)]
    tested = [r for r in rows if r["sef_null_tree"] is not None]
    log("\n%d positions, %d given a tree null, %d calibration failures, %d "
        "rates at a grid edge"
        % (len(rows), len(tested), failed_cal, edge_hits[0]))
    if tested and failed_cal > len(tested) / 2:
        raise P.Fatal("rate calibration failed for %d of %d columns; the grid "
                      "does not span the observed entropies and the null is not "
                      "usable" % (failed_cal, len(tested)))

    log("\nmean null sef, the two nulls compared over the %d tested columns"
        % len(tested))
    pn = tested
    log("  observed sef                     %.3f"
        % float(np.mean([r["sef"] for r in pn])))
    log("  tree null mean sef               %.3f"
        % float(np.mean([r["sef_null_tree"] for r in pn])))
    log("  chance-corrected under permutation %.3f"
        % float(np.mean([r["sef_adj_perm"] for r in pn])))
    log("  chance-corrected under the tree    %.3f"
        % float(np.mean([r["sef_adj_tree"] for r in pn])))

    log("\nclassification under each null, over the %d tested columns"
        % len(tested))
    log("  call            permutation   tree")
    for c in ("diagnostic", "tolerant", "indeterminate"):
        a_ = sum(1 for r in tested if r["call_perm"] == c)
        b_ = sum(1 for r in tested if r["call_tree"] == c)
        log("  %-14s %7d %5.1f%% %7d %5.1f%%"
            % (c, a_, 100 * a_ / len(tested), b_, 100 * b_ / len(tested)))

    if DESIGNED:
        log("\nthe positions Table 1 names individually")
    log("  pos wt  set              sef   perm adj  p_perm   tree null  tree adj"
        "  p_tree   perm call      tree call")
    for pos in list(DESIGNED) + list(HELD_BACK) + list(NATURAL):
        r = next(x for x in rows if x["position"] == pos)
        log("  %-4d %s  %-15s %5s  %7s  %6s  %9s  %8s  %6s   %-13s %s"
            % (r["position"], r["wt"], r["set"], r["sef"], r["sef_adj_perm"],
               None if r["p_perm"] is None else round(r["p_perm"], 4),
               r["sef_null_tree"], r["sef_adj_tree"], r["p_tree"],
               r["call_perm"], r["call_tree"]))

    if FUNCTIONAL:
        ctrl = [r for r in rows if r["set"] == "functional"]
        bad = [r["position"] for r in ctrl if r["call_perm"] != "conserved"
               or r["call_tree"] not in ("conserved", "")]
        if bad:
            raise P.Fatal("control positions %s are no longer conserved under "
                          "both nulls; the run is not trustworthy" % bad)
        log("\n  control PASS: the ten catalytic and cofactor positions stay "
            "conserved under both nulls")

    os.makedirs(a.outdir, exist_ok=True)
    cols = ["position", "wt", "set", "n", "gap_frac", "h_total", "sef",
            "sef_adj_perm", "p_perm", "call_perm", "rate", "rate_at_grid_edge",
            "parsimony_obs", "parsimony_null", "null_h_mean",
            "sef_null_tree", "sef_adj_tree", "p_tree", "call_tree",
            "calibrated"]
    out = os.path.join(a.outdir, "tree_null_%s.csv" % a.name)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log("\nwrote %s (%d rows)" % (out, len(rows)))
    logp = os.path.join(a.outdir, "tree_null_%s_log.txt" % a.name)
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except P.Fatal as exc:
        sys.exit("FATAL: %s" % exc)
