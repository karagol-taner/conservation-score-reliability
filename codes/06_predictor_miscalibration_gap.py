#!/usr/bin/env python3

import argparse
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

import partition as P
import pandas as pd

GAP_HI = 0.50
MIN_RESIDUES = 5000
HUMSAVAR_ROW = re.compile(
    r"^(\S+)\s+([A-Z0-9]+)\s+(VAR_\d+)\s+p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\s+"
    r"(LB/B|LP/P|US)\b")
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}


class Fatal(RuntimeError):
    pass


def rho(x, y):

    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def partial_rho(x, y, z):

    rxy, rxz, ryz = rho(x, y), rho(x, z), rho(y, z)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    if den <= 1e-12:
        return float("nan")
    return (rxy - rxz * ryz) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn", default=None)
    ap.add_argument("--positions", default=os.path.join(
        "..", "data", "gap_inflation_positions.csv.gz"))


    ap.add_argument("--humsavar", default=P.humsavar_path())
    ap.add_argument("--outdir", default=os.path.join("..", "data"))
    a = ap.parse_args()
    if a.dyn is None:
        a.dyn = P.dynamics_dir()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("06_predictor_miscalibration_gap.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("Does the gap flag predict AlphaMissense error, and in which direction?")
    log("The prediction is a fork and both branches are informative; see the")
    log("script docstring, written before these numbers existed.")
    log("")
    log("gap threshold %.2f, correlations are Pearson on ranks" % GAP_HI)

    if not os.path.exists(a.positions):
        raise Fatal("no position table at %s; run script 03 first" % a.positions)
    pos = pd.read_csv(a.positions)
    need = {"cohort", "protein_id", "i", "conservation", "gap_frac",
            "cons_ungapped"}
    if not need.issubset(pos.columns):
        raise Fatal("position table lacks %s" % sorted(need - set(pos.columns)))

    summary = []

    def rec(cohort, measure, value, n=None, note=""):
        summary.append(dict(cohort=cohort, measure=measure, n=n,
                            value=None if value is None or (isinstance(value, float)
                                                            and np.isnan(value))
                            else round(float(value), 4), note=note))

    frames = []
    for cohort in ("data-human", "data-alphahelical"):
        master = os.path.join(a.dyn, cohort, "data_tsv",
                              "All_Proteins_AM_Combined_Master.tsv")
        if not os.path.exists(master):
            raise Fatal("no AlphaMissense master at %s" % master)
        m = pd.read_csv(master, sep="\t").rename(
            columns={"Protein_ID": "protein_id"})
        if "mean_am_pathogenicity" not in m.columns:
            raise Fatal("%s has no mean_am_pathogenicity column" % master)
        sub = pos[pos["cohort"] == cohort]
        j = m.merge(sub, on=["protein_id", "i"], how="inner",
                    suffixes=("", "_col"))
        j = j.dropna(subset=["mean_am_pathogenicity", "gap_frac",
                             "cons_ungapped"])
        j["cohort"] = cohort
        log("\n%s: %d residues with both an AlphaMissense mean and a column"
            % (cohort, len(j)))
        frames.append(j)
    d = pd.concat(frames, ignore_index=True)
    if len(d) < MIN_RESIDUES:
        raise Fatal("only %d residues joined, below the stated minimum of %d"
                    % (len(d), MIN_RESIDUES))

    log("\n%d residues over %d proteins in total"
        % (len(d), d["protein_id"].nunique()))


    log("\nwhich conservation definition tracks AlphaMissense better")
    log("  cohort            n      rho(reported, AM)  rho(gap-free, AM)  "
        "rho(gap, AM)")
    for cohort in ("data-human", "data-alphahelical", "both"):
        s = d if cohort == "both" else d[d["cohort"] == cohort]
        r_rep = rho(s["conservation"], s["mean_am_pathogenicity"])
        r_free = rho(s["cons_ungapped"], s["mean_am_pathogenicity"])
        r_gap = rho(s["gap_frac"], s["mean_am_pathogenicity"])
        log("  %-16s %6d  %17.3f  %17.3f  %12.3f"
            % (cohort, len(s), r_rep, r_free, r_gap))
        rec(cohort, "rho_reported_conservation_vs_AM", r_rep, len(s))
        rec(cohort, "rho_gapfree_conservation_vs_AM", r_free, len(s))
        rec(cohort, "rho_gap_vs_AM", r_gap, len(s))

    log("\npartial correlation of gap fraction with AlphaMissense, holding the")
    log("gap-free conservation fixed, which separates gappiness from constraint")
    for cohort in ("data-human", "data-alphahelical", "both"):
        s = d if cohort == "both" else d[d["cohort"] == cohort]
        pr = partial_rho(s["gap_frac"], s["mean_am_pathogenicity"],
                         s["cons_ungapped"])
        log("  %-16s %6d  partial rho %7.3f" % (cohort, len(s), pr))
        rec(cohort, "partial_rho_gap_vs_AM_given_gapfree", pr, len(s))

    log("\nmean AlphaMissense by gap decile")
    d["gap_decile"] = (d["gap_frac"] * 10).clip(0, 9).astype(int)
    g = d.groupby("gap_decile").agg(n=("mean_am_pathogenicity", "size"),
                                    am=("mean_am_pathogenicity", "mean"),
                                    cons=("conservation", "mean"),
                                    free=("cons_ungapped", "mean"))
    for dec, row in g.iterrows():
        log("  gap %2d0-%2d0%%  n=%6d  AlphaMissense %.3f  reported cons %.3f  "
            "gap-free cons %.3f"
            % (dec, dec + 1, row["n"], row["am"], row["cons"], row["free"]))
        rec("both", "mean_AM_gap_decile_%d" % dec, row["am"], int(row["n"]))

    hi = d[d["gap_frac"] > GAP_HI]
    lo = d[d["gap_frac"] <= GAP_HI]
    log("\n  above %.0f%% gap: n=%d, mean AlphaMissense %.3f, mean reported "
        "conservation %.3f" % (100 * GAP_HI, len(hi),
                               hi["mean_am_pathogenicity"].mean(),
                               hi["conservation"].mean()))
    log("  at or below:     n=%d, mean AlphaMissense %.3f, mean reported "
        "conservation %.3f" % (len(lo), lo["mean_am_pathogenicity"].mean(),
                               lo["conservation"].mean()))
    rec("both", "mean_AM_gappy", hi["mean_am_pathogenicity"].mean(), len(hi))
    rec("both", "mean_AM_not_gappy", lo["mean_am_pathogenicity"].mean(), len(lo))


    if a.humsavar:
        if not os.path.exists(a.humsavar):
            raise Fatal("no humsavar at %s" % a.humsavar)
        accs = set(d["UniProt_ID"].dropna().unique())
        log("\nlabelled arm: humsavar variants in the %d cohort accessions"
            % len(accs))
        rows = []
        with open(a.humsavar, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                mm = HUMSAVAR_ROW.match(line)
                if not mm:
                    continue
                gene, acc, vid, aa1, posn, aa2, cls = mm.groups()
                if acc not in accs:
                    continue
                rows.append(dict(gene=gene, UniProt_ID=acc, i=int(posn),
                                 wt=AA3.get(aa1.upper()), mut=AA3.get(aa2.upper()),
                                 classification=cls))
        if not rows:
            log("  none matched")
        else:
            hv = pd.DataFrame(rows)
            key = d[["UniProt_ID", "i", "gap_frac", "conservation",
                     "cons_ungapped", "mean_am_pathogenicity", "A_i"]].copy()
            j = hv.merge(key, on=["UniProt_ID", "i"], how="inner")
            bad = j[j["wt"] != j["A_i"]]
            log("  %d variants matched a scored position, %d with a wild-type "
                "residue disagreeing with the cohort sequence"
                % (len(j), len(bad)))
            if len(bad):
                raise Fatal("%d humsavar variants disagree with the cohort "
                            "sequence at their position; the numbering does not "
                            "agree and no labelled table may be built" % len(bad))
            cc = Counter(j["classification"])
            log("  classifications: " + ", ".join("%s %d" % kv
                                                  for kv in sorted(cc.items())))
            log("\n  mean AlphaMissense by label and gap status")
            log("    label  gappy  n     mean AM   mean reported cons")
            for cls in ("LB/B", "LP/P", "US"):
                for flag in (True, False):
                    s = j[(j["classification"] == cls)
                          & ((j["gap_frac"] > GAP_HI) == flag)]
                    if not len(s):
                        continue
                    log("    %-6s %-6s %-5d %7.3f   %7.3f"
                        % (cls, flag, len(s), s["mean_am_pathogenicity"].mean(),
                           s["conservation"].mean()))
                    rec("both", "mean_AM_%s_gappy_%s" % (cls.replace("/", ""),
                                                         flag),
                        s["mean_am_pathogenicity"].mean(), len(s))
            ben = j[j["classification"] == "LB/B"]
            pat = j[j["classification"] == "LP/P"]
            for name, s in (("benign-side", ben), ("pathogenic-side", pat)):
                if len(s) > 20:
                    r = rho(s["gap_frac"], s["mean_am_pathogenicity"])
                    log("  rho(gap, AM) among %s variants: %.3f over %d"
                        % (name, r, len(s)))
                    rec("both", "rho_gap_vs_AM_%s" % name.split("-")[0], r,
                        len(s))
                else:
                    log("  %s variants: %d, too few for a correlation"
                        % (name, len(s)))

    os.makedirs(a.outdir, exist_ok=True)
    outs = os.path.join(a.outdir, "miscalibration_gap_summary.csv")
    pd.DataFrame(summary).to_csv(outs, index=False)
    log("\nwrote %s (%d rows)" % (outs, len(summary)))
    logp = os.path.join(a.outdir, "miscalibration_gap_log.txt")
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        sys.exit("FATAL: %s" % exc)
