#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

import partition as P

MIN_EFFECTIVE = 5
SISTER_FIX = 1.0
GRADES = 9
SEED = 20261007

HUMSAVAR_ROW = re.compile(
    r"^(\S+)\s+([A-Z0-9]+)\s+(VAR_\d+)\s+p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\s+"
    r"(LB/B|LP/P|US)\b")
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}


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


def accession(header):
    m = re.search(r"\|([A-Z0-9]+)\|", header)
    return m.group(1) if m else ""


def own_positions(row):

    out, pos = {}, 0
    for col, ch in enumerate(row):
        if ch != P.GAP:
            pos += 1
            out[pos] = col
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--ref", required=True,
                    help="substring identifying the reference header")
    ap.add_argument("--own", required=True,
                    help="subfamily label the reference belongs to")
    ap.add_argument("--subfamilies", required=True)


    ap.add_argument("--humsavar", default=P.humsavar_path())
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=os.path.join("..", "data"))
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()

    tag = a.tag or a.own.lower()
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("05_reliability_panel.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("Per-position reliability of a conservation score, with the direction of")
    log("the error where there is one.")
    log("")
    log("thresholds: gap %.2f, effective comparators %d, entropy floor %.2f "
        "bits," % (P.MAX_GAP_FRAC, MIN_EFFECTIVE, P.H_MIN))
    log("            sef_adj %.2f with p <= %.3f, sister fixed at %.2f identity,"
        % (P.SEF_HI, P.P_MAX, SISTER_FIX))
    log("            n_perm %d, seed %d" % (a.n_perm, SEED))

    subs = parse_subfamilies(a.subfamilies)
    if a.own not in subs:
        raise P.Fatal("--own %r is not among the subfamily labels %s"
                      % (a.own, sorted(subs)))

    fam = P.read_fasta(a.fasta)
    ref_header = next((k for k in fam if a.ref in k), None)
    if ref_header is None:
        raise P.Fatal("no header containing %r in %s" % (a.ref, a.fasta))
    ref_seq = fam[ref_header]

    labels, unlabelled = P.assign_labels(fam, subs)
    if labels.get(ref_header) != a.own:
        raise P.Fatal("the reference labelled %r, not %r"
                      % (labels.get(ref_header), a.own))
    sizes = Counter(labels.values())
    log("\nreference %s, %d residues" % (accession(ref_header) or a.ref,
                                         len(ref_seq)))
    for lab in sorted(sizes):
        log("  %-6s %2d sequences" % (lab, sizes[lab]))
    log("  %d sequences carry no label and are excluded" % len(unlabelled))
    own_h = [h for h, l in labels.items() if l == a.own]
    other_h = [h for h, l in labels.items() if l != a.own]
    P.check_disjoint(own_h, other_h)

    sub = {k: v for k, v in fam.items() if k in labels}
    msa = P.align(sub)
    pos_to_col = P.map_positions(msa, ref_header, ref_seq)
    log("  aligned %d sequences, width %d, reference row reconstructs its input"
        % (len(sub), len(next(iter(msa.values())))))

    sister_labels = [l for l in sizes if l != a.own]
    rng = np.random.default_rng(SEED)

    rows = []
    for pos, col in pos_to_col.items():
        wt = ref_seq[pos - 1]
        res, labs, gap_frac, dropped = P.column_set(msa, col, ref_header, labels)
        stat = P.column_partition(res, labs, rng, n_perm=a.n_perm)
        call = P.classify(stat)
        f_fam, n_fam, f_own, n_own = P.identities(msa, col, ref_header, wt,
                                                  labels, a.own)
        n_eff = stat["n"]

        if gap_frac > P.MAX_GAP_FRAC:
            verdict, sign = "gap_inflated", "reported conservation too high"
        elif n_eff < MIN_EFFECTIVE:
            verdict, sign = "depth_starved", "no estimate"
        elif call == "diagnostic":
            verdict, sign = "clade_partitioned", "family conservation too low"
        else:
            verdict, sign = "ok", ""


        sisters = []
        for lab in sister_labels:
            vals = [msa[h][col] for h in msa
                    if labels.get(h) == lab and msa[h][col] != P.GAP]
            if not vals:
                continue
            top, n_top = Counter(vals).most_common(1)[0]
            if n_top / len(vals) >= SISTER_FIX and top != wt:
                sisters.append("%s>%s:%s" % (wt, top, lab))

        rows.append(dict(
            position=pos, wt=wt, verdict=verdict, error_sign=sign,
            n_effective=n_eff, gap_frac=round(gap_frac, 4),
            dropped_subfamilies=";".join(dropped),
            h_total=round(stat["h_total"], 4),
            sef=None if stat["sef"] is None else round(stat["sef"], 4),
            sef_adj=None if stat["sef_adj"] is None else round(stat["sef_adj"], 4),
            p_perm=stat["p_perm"], call=call,
            f_family=f_fam, n_family=n_fam, f_own=f_own, n_own=n_own,
            inverted=P.inverted(f_fam, f_own),
            sister_substitutions=";".join(sisters), n_sister=len(sisters),
            distribution=P.distribution(res)))

    v = Counter(r["verdict"] for r in rows)
    log("\nverdicts over %d reference positions" % len(rows))
    for k in ("ok", "gap_inflated", "depth_starved", "clade_partitioned"):
        log("  %-18s %4d  %5.1f%%" % (k, v[k], 100 * v[k] / len(rows)))
    flagged = len(rows) - v["ok"]
    log("  %d positions carry a flag, %.1f%%" % (flagged,
                                                 100 * flagged / len(rows)))

    withs = [r for r in rows if r["n_sister"]]
    log("\n%d positions carry at least one fixed sister state, %.1f%%"
        % (len(withs), 100 * len(withs) / len(rows)))
    cp = [r for r in withs if r["verdict"] == "clade_partitioned"]
    log("  of those, %d are clade-partitioned, which is the set the "
        "sister-state test takes" % len(cp))
    subst = Counter(s.split(":")[0] for r in withs
                    for s in r["sister_substitutions"].split(";") if s)
    log("  the ten commonest ordered sister substitutions:")
    for s, c in subst.most_common(10):
        log("    %-8s %d" % (s, c))


    variants = []
    if a.humsavar:
        if not os.path.exists(a.humsavar):
            raise P.Fatal("no humsavar at %s" % a.humsavar)
        acc_to_header = {}
        for h in msa:
            acc = accession(h)
            if acc:
                acc_to_header.setdefault(acc, h)
        own_maps = {h: own_positions(msa[h]) for h in msa}
        col_to_refpos = {c: p for p, c in pos_to_col.items()}
        by_pos = {r["position"]: r for r in rows}

        n_seen = 0
        with open(a.humsavar, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = HUMSAVAR_ROW.match(line)
                if not m:
                    continue
                gene, acc, vid, aa1, posn, aa2, cls = m.groups()
                if acc not in acc_to_header:
                    continue
                h = acc_to_header[acc]
                wt1, mut = AA3.get(aa1.upper()), AA3.get(aa2.upper())
                posn = int(posn)
                seq = fam[h]
                if wt1 is None or mut is None:
                    continue
                n_seen += 1
                if posn > len(seq) or seq[posn - 1] != wt1:
                    n_bad += 1
                    raise P.Fatal(
                        "humsavar %s %s %s says position %d is %s but the "
                        "sequence carries %s; the numbering does not agree and "
                        "no variant table may be built"
                        % (gene, acc, vid, posn,
                           wt1, seq[posn - 1] if posn <= len(seq) else "nothing"))
                col = own_maps[h].get(posn)
                refpos = col_to_refpos.get(col) if col is not None else None
                r = by_pos.get(refpos) if refpos else None
                sister_hit, sister_subs = "", []
                if r:
                    for s in r["sister_substitutions"].split(";"):
                        if s and s.split(":")[0].endswith(">" + mut):
                            sister_hit = s
                            sister_subs.append(s.split(":")[1])
                    sister_hit = ";".join(sister_subs)
                variants.append(dict(
                    gene=gene, accession=acc, variant_id=vid,
                    own_position=posn, wt=wt1, mut=mut, classification=cls,
                    subfamily=labels.get(h, ""),
                    ref_position=refpos or "", mapped=bool(r),
                    verdict=r["verdict"] if r else "",
                    sef_adj=r["sef_adj"] if r else "",
                    f_own=r["f_own"] if r else "",
                    sister_state=bool(sister_hit),
                    sister_residue=mut if sister_hit else "",
                    sister_subfamilies=sister_hit))
        log("\nhumsavar cross-reference: %d variants in family members, %d "
            "mapped to a reference column" % (n_seen, sum(1 for x in variants
                                                          if x["mapped"])))
        log("  every variant's wild-type residue agreed with its sequence")
        cls_c = Counter(x["classification"] for x in variants)
        log("  classifications: " + ", ".join("%s %d" % kv
                                              for kv in sorted(cls_c.items())))
        ss = [x for x in variants if x["sister_state"]]
        log("  %d variants substitute to a fixed sister-subfamily residue"
            % len(ss))
        if ss:
            log("    the mutant residue is the fixed state of the named "
                "subfamily")
            log("    gene    variant        class  ref pos  verdict            "
                "fixed in")
            for x in ss:
                log("    %-7s %s%d%s%s  %-5s  %7s  %-18s %s"
                    % (x["gene"], x["wt"], x["own_position"], x["mut"],
                       " " * max(0, 6 - len(str(x["own_position"]))),
                       x["classification"], x["ref_position"], x["verdict"],
                       x["sister_subfamilies"]))
            tab = Counter((x["sister_state"], x["classification"])
                          for x in variants if x["mapped"])
            log("\n  sister-state against classification, mapped variants only")
            log("    sister  LB/B  LP/P  US")
            for flag in (True, False):
                log("    %-6s %5d %5d %3d"
                    % (flag, tab[(flag, "LB/B")], tab[(flag, "LP/P")],
                       tab[(flag, "US")]))
            log("  n is small here, so this table reports the annotation and "
                "not a")
            log("  result; the pooled test runs over all families in "
                "09_sister_state_test.py.")


    os.makedirs(a.outdir, exist_ok=True)
    cols = ["position", "wt", "verdict", "error_sign", "n_effective",
            "gap_frac", "dropped_subfamilies", "h_total", "sef", "sef_adj",
            "p_perm", "call", "f_family", "n_family", "f_own", "n_own",
            "inverted", "sister_substitutions", "n_sister", "distribution"]
    out = os.path.join(a.outdir, "reliability_panel_%s.csv" % tag)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log("\nwrote %s (%d rows)" % (out, len(rows)))

    if variants:
        vcols = ["gene", "accession", "variant_id", "subfamily", "own_position",
                 "wt", "mut", "classification", "ref_position", "mapped",
                 "verdict", "sef_adj", "f_own", "sister_state",
                 "sister_residue", "sister_subfamilies"]
        outv = os.path.join(a.outdir, "reliability_variants_%s.csv" % tag)
        with open(outv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=vcols, extrasaction="ignore")
            w.writeheader()
            w.writerows(variants)
        log("wrote %s (%d rows)" % (outv, len(variants)))

    logp = os.path.join(a.outdir, "reliability_panel_%s_log.txt" % tag)
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except P.Fatal as exc:
        sys.exit("FATAL: %s" % exc)
