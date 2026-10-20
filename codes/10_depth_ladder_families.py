#!/usr/bin/env python3

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import numpy as np

import partition as P

HUMAN = "Homo sapiens"
SEED = 20261007
VARIABLE_MAX = P.INV_FAM_MAX
CONSERVED_MIN = P.INV_OWN_MIN


PEROXIDASE_EXPECTED = {
    "D1_paralogues": dict(mean_abs_shift=2.0789, shift_ge3=0.328, flip=0.0735),
    "D3_subfamily": dict(mean_abs_shift=1.7957, shift_ge3=0.2796, flip=0.0627),
}


def parse_subfamilies(spec):

    out = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        if "=" not in part:
            raise P.Fatal("subfamily spec %r is not label=pattern" % part)
        lab, pat = part.split("=", 1)
        if not pat:
            raise P.Fatal("subfamily %r has an empty pattern" % lab.strip())
        out.setdefault(lab.strip(), []).append(pat)
    return {k: tuple(v) for k, v in out.items()}


def select(fam, ref_header, labels, own, which):
    out = []
    for h in fam:
        if h == ref_header:
            continue
        lab = labels.get(h)
        if which == "D1_paralogues":
            if lab is not None and lab != own and P.organism(h) == HUMAN:
                out.append(h)
        elif which == "D2_orthologues":
            if lab == own:
                out.append(h)
        elif which == "D3_subfamily":
            if lab is not None:
                out.append(h)
        elif which == "D4_family":
            out.append(h)
        else:
            raise P.Fatal("unknown depth %r" % which)
    return out


def depth_profile(fam, ref_header, ref_seq, comparators, log):
    seqs = {ref_header: ref_seq}
    for h in comparators:
        seqs[h] = fam[h]
    msa = P.align(seqs)
    pos_to_col = P.map_positions(msa, ref_header, ref_seq)
    others = [h for h in msa if h != ref_header]
    prof = {}
    for pos, col in pos_to_col.items():
        wt = ref_seq[pos - 1]
        chars = [msa[h][col] for h in others]
        ungapped = [c for c in chars if c != "-"]
        prof[pos] = dict(
            identity=(sum(1 for c in ungapped if c == wt) / len(ungapped)
                      if ungapped else None),
            gap_frac=1.0 - len(ungapped) / len(chars) if chars else 1.0,
            n=len(ungapped))
    return prof, msa, pos_to_col


def grades(use, key):

    vals = sorted((r[key], r["position"]) for r in use)
    out, n = {}, len(vals)
    for rank, (_v, pos) in enumerate(vals):
        out[pos] = min(9, 1 + int(9 * rank / n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--ref", required=True, help="substring of the reference header")
    ap.add_argument("--own", required=True, help="the reference's subfamily label")
    ap.add_argument("--subfamilies", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--min-paralogues", type=int, default=2)
    ap.add_argument("--min-orthologues", type=int, default=3)
    ap.add_argument("--min-subfamily", type=int, default=8)
    ap.add_argument("--outdir", default=os.path.join("..", "data"))
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("10_depth_ladder_families.py  run %s"
        % datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("How far does a position move when only the comparator set changes?")
    log("family %s, reference %s, own subfamily %s" % (a.tag, a.ref, a.own))

    fam = P.read_fasta(a.fasta)
    subs = parse_subfamilies(a.subfamilies)
    labels, unlabelled = P.assign_labels(fam, subs)
    ref_header = next((k for k in fam if a.ref in k), None)
    if ref_header is None:
        raise P.Fatal("no header containing %r in %s" % (a.ref, a.fasta))
    ref_seq = fam[ref_header]
    if labels.get(ref_header) != a.own:
        raise P.Fatal("reference %s is labelled %r, not the stated own subfamily %r"
                      % (ref_header, labels.get(ref_header), a.own))
    log("  %d sequences, %d labelled, %d outside the labelled subfamilies"
        % (len(fam), len(labels), len(fam) - len(labels)))

    depths = ["D1_paralogues", "D2_orthologues", "D3_subfamily"]
    if len(fam) - len(labels) > 0:
        depths.append("D4_family")
        log("  the file holds unlabelled members, so a family depth is added")
    else:
        log("  every sequence is labelled, so there is no depth beyond the "
            "subfamily set and none is invented")

    sets, mins = {}, {"D1_paralogues": a.min_paralogues,
                      "D2_orthologues": a.min_orthologues,
                      "D3_subfamily": a.min_subfamily,
                      "D4_family": a.min_subfamily}
    for d in depths:
        sets[d] = select(fam, ref_header, labels, a.own, d)
        log("  %-16s %4d comparators" % (d, len(sets[d])))
        if len(sets[d]) < mins[d]:
            raise P.Fatal("%s has %d comparators, below the stated minimum %d"
                          % (d, len(sets[d]), mins[d]))
    P.check_disjoint(sets["D2_orthologues"], sets["D1_paralogues"])

    prof = {}
    for d in depths:
        prof[d], _msa, _cols = depth_profile(fam, ref_header, ref_seq, sets[d], log)

    rows = []
    for pos in sorted(prof["D2_orthologues"]):
        r = dict(position=pos, wt=ref_seq[pos - 1])
        for d in depths:
            p = prof[d][pos]
            r["id_" + d] = p["identity"]
            r["gap_" + d] = p["gap_frac"]
            r["n_" + d] = p["n"]
        rows.append(r)

    use = [r for r in rows
           if all(r["gap_" + d] <= P.MAX_GAP_FRAC for d in depths)
           and all(r["id_" + d] is not None for d in depths)]
    log("\n%d of %d positions pass the gap filter at every depth"
        % (len(use), len(rows)))
    if not use:
        raise P.Fatal("no position passes the gap filter at every depth")

    grade = {d: grades(use, "id_" + d) for d in depths}
    for r in use:
        for d in depths:
            r["grade_" + d] = grade[d][r["position"]]

    log("\ngrade movement against orthologue depth, ConSurf units")
    log("  depth             mean |shift|  >=3 grades   variable to conserved"
        "   inverted")
    summary = []
    for d in [x for x in depths if x != "D2_orthologues"]:
        shifts = [abs(r["grade_" + d] - r["grade_D2_orthologues"]) for r in use]
        mean_shift = float(np.mean(shifts))
        ge3 = sum(1 for s in shifts if s >= 3) / len(use)
        flip = sum(1 for r in use
                   if r["grade_" + d] <= 3 and r["grade_D2_orthologues"] >= 7) / len(use)
        inv = sum(1 for r in use
                  if r["id_" + d] < VARIABLE_MAX
                  and r["id_D2_orthologues"] >= CONSERVED_MIN) / len(use)
        log("  %-16s %12.4f  %9.1f%%  %20.1f%%  %8.1f%%"
            % (d, mean_shift, 100 * ge3, 100 * flip, 100 * inv))
        summary.append(dict(family=a.tag, depth=d, n_positions=len(use),
                            comparators=len(sets[d]),
                            mean_abs_shift=round(mean_shift, 4),
                            shift_ge3=round(ge3, 4), flip=round(flip, 4),
                            inversion_rate=round(inv, 4)))


    if a.tag == "peroxidase":
        for row in summary:
            want = PEROXIDASE_EXPECTED.get(row["depth"])
            if not want:
                continue
            for k, v in want.items():
                if abs(row[k] - v) > 0.0006:
                    raise P.Fatal(
                        "peroxidase %s %s is %.4f here and %.4f in "
                        "02_depth_ladder_peroxidase.py; the generalisation has "
                        "changed the measurement" % (row["depth"], k, row[k], v))
        log("\ncontrol PASS: reproduces every headline number "
            "02_depth_ladder_peroxidase.py reports")

    outdir = a.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    per = os.path.join(outdir, "depth_ladder_fam_%s.csv" % a.tag)
    with open(per, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(use[0].keys()))
        w.writeheader()
        w.writerows(use)

    agg = os.path.join(outdir, "depth_ladder_families.csv")
    old = []
    if os.path.exists(agg):
        with open(agg, newline="") as fh:
            old = [r for r in csv.DictReader(fh) if r["family"] != a.tag]
    with open(agg, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for r in old:
            w.writerow({k: r.get(k, "") for k in summary[0]})
        w.writerows(summary)

    with open(os.path.join(outdir, "depth_ladder_fam_%s_log.txt" % a.tag), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote %s (%d rows) and updated %s" % (per, len(use), agg))


if __name__ == "__main__":
    try:
        main()
    except P.Fatal as exc:
        print("\nFATAL: %s" % exc, file=sys.stderr)
        sys.exit(1)
