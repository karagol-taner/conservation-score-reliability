#!/usr/bin/env python3

import argparse
import glob
import gzip
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

import partition as P
import pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")
GAP = "-"
GAP_HI = 0.50
LOG2_21 = np.log2(21)
LOG2_20 = np.log2(20)
AM_PATHOGENIC = 0.90
AM_BENIGN = 0.20
TOP_PCT = 0.01
TOP_DECILE = 0.10
BITSCORE = re.compile(r"_b(0\.\d)_")

VARIANTS = {"A_as_published": dict(filter_gaps=False, ungapped=False),
            "B_gap_filter": dict(filter_gaps=True, ungapped=False),
            "C_gap_corrected": dict(filter_gaps=False, ungapped=True),
            "D_both": dict(filter_gaps=True, ungapped=True)}


DISCOVERY = [("NARS1", "p.Thr17Met", "O43776", 17, 5.78, 7.96, 0.09),
             ("NARS1", "p.Lys60Thr", "O43776", 60, 4.01, 7.82, 0.53),
             ("PGK1", "p.Lys30Thr", "P00558", 30, 3.26, 0.62, 0.47),
             ("SPRTN", "p.Tyr117Cys", "Q9H040", 117, 0.53, 6.18, 0.47)]


class Fatal(RuntimeError):
    pass


def spearman(x, y):

    return x.rank().corr(y.rank())


def auc(scores, positive):

    s = pd.Series(np.asarray(scores, dtype=float))
    y = np.asarray(positive, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None, npos, nneg
    r = s.rank().to_numpy()
    return (r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg), npos, nneg


def load_columns(root, cohort):

    pat = os.path.join(root, cohort, "EVcouplings", "align_frequencies",
                       "*_frequencies.csv")
    files = sorted(glob.glob(pat))
    if not files:
        raise Fatal("no frequency files under %s" % pat)
    frames = []
    for path in files:
        base = os.path.basename(path)
        d = pd.read_csv(path)
        present = [c for c in AA if c in d.columns]
        if len(present) != 20 or GAP not in d.columns:
            raise Fatal("%s does not carry the gap column and all 20 residue "
                        "columns" % base)
        f20 = d[present].to_numpy(float)
        mass = f20.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(mass[:, None] > 0, f20 / mass[:, None], 0.0)
            h20 = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
        f21 = np.column_stack([d[GAP].to_numpy(float), f20])
        s21 = f21.sum(axis=1, keepdims=True)
        q = np.divide(f21, s21, out=np.zeros_like(f21), where=s21 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h21 = -np.nansum(np.where(q > 0, q * np.log2(q), 0.0), axis=1)
        m = BITSCORE.search(base)
        frames.append(pd.DataFrame(dict(
            protein_id=base.split("_")[0], i=d["i"].to_numpy(),
            bitscore=float(m.group(1)) if m else np.nan,
            gap_frac=d[GAP].to_numpy(float),
            cons_reported=d["conservation"].to_numpy(float),
            cons_recovered=1.0 - h21 / LOG2_21,
            cons_ungapped=1.0 - h20 / LOG2_20,
            ungapped_mass=mass)))
    return pd.concat(frames, ignore_index=True), len(files)


def rescale(group, col):

    v = group[col].to_numpy(float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or hi - lo <= 0:
        return pd.Series(np.zeros_like(v), index=group.index)
    return pd.Series((v - lo) / (hi - lo) * 10.0, index=group.index)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dyn", default=None)
    ap.add_argument("--outdir", default=os.path.join("..", "data"))
    a = ap.parse_args()
    if a.dyn is None:
        a.dyn = P.dynamics_dir()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("04_gap_survival_dynamics_aware.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("Does the published dynamics-aware result change when the gap symbol is")
    log("removed from the conservation alphabet?")
    log("")
    log("gap threshold %.2f, AlphaMissense labels >%.2f positive and <%.2f "
        "negative" % (GAP_HI, AM_PATHOGENIC, AM_BENIGN))

    if not os.path.isdir(a.dyn):
        raise Fatal("no dynamics-aware folder at %s" % a.dyn)

    summary = []

    def rec(cohort, variant, measure, value, n=None, note=""):
        summary.append(dict(cohort=cohort, variant=variant, measure=measure,
                            n=n, value=None if value is None
                            else round(float(value), 4), note=note))

    allscores = []
    for cohort in ("data-human", "data-alphahelical"):
        log("\n" + "=" * 72)
        log(cohort)
        log("=" * 72)
        cols, nfiles = load_columns(a.dyn, cohort)
        master = os.path.join(a.dyn, cohort, "data_tsv",
                              "All_Proteins_AM_Combined_Master.tsv")
        if not os.path.exists(master):
            raise Fatal("no AlphaMissense master at %s" % master)
        m = pd.read_csv(master, sep="\t").rename(
            columns={"Protein_ID": "protein_id"})
        d = m.merge(cols, on=["protein_id", "i"], how="inner",
                    suffixes=("", "_col"))
        log("  %d frequency files, %d master residues, %d matched to a column"
            % (nfiles, len(m), len(d)))
        if not len(d):
            raise Fatal("no residues matched a column in %s" % cohort)


        err = (d["cons_reported"] - d["cons_recovered"]).abs().max()
        log("  conservation reproduced as 1 - H/log2(21) over the gap-inclusive")
        log("  alphabet to within %.4f, so the gap is a conserved character by"
            % err)
        log("  construction and the ungapped recomputation is the matched "
            "control")
        if err > 0.05:
            raise Fatal("conservation column is not reproduced by "
                        "1 - H/log2(21) (max error %.4f); the mechanism stated "
                        "in this script is wrong and nothing below is valid"
                        % err)
        rec(cohort, "-", "max_error_conservation_recovery", err, len(d))

        gappy = d["gap_frac"] > GAP_HI
        log("  %d of %d residues (%.1f%%) sit on columns above %.0f%% gap"
            % (gappy.sum(), len(d), 100 * gappy.mean(), 100 * GAP_HI))
        rec(cohort, "-", "share_residues_on_gappy_columns", gappy.mean(),
            int(gappy.sum()))

        for vname, cfg in VARIANTS.items():
            v = d[~gappy].copy() if cfg["filter_gaps"] else d.copy()
            v["cons_used"] = (v["cons_ungapped"] if cfg["ungapped"]
                              else v["cons_reported"])

            v["nd"] = v.groupby("protein_id", group_keys=False).apply(
                lambda g: (g["Avg_RMSF"] - g["Avg_RMSF"].min())
                / (g["Avg_RMSF"].max() - g["Avg_RMSF"].min())
                if g["Avg_RMSF"].max() > g["Avg_RMSF"].min()
                else pd.Series(np.zeros(len(g)), index=g.index))
            v["raw_dcs"] = v["cons_used"] * v["nd"]
            v["raw_rcs"] = v["cons_used"] * (1.0 - v["nd"])
            for raw, out in (("raw_dcs", "dcs10"), ("raw_rcs", "rcs10")):
                v[out] = v.groupby("protein_id", group_keys=False).apply(
                    lambda g, c=raw: rescale(g, c))

            log("\n  %s   n = %d residues, %d proteins"
                % (vname, len(v), v["protein_id"].nunique()))
            r_dr = spearman(v["dcs10"], v["rcs10"])
            r_rc = spearman(v["rcs10"], v["cons_used"])
            r_dd = spearman(v["dcs10"], v["nd"])
            log("    rho(DCS, RCS) %7.3f    rho(RCS, conservation) %7.3f    "
                "rho(DCS, dynamics) %7.3f" % (r_dr, r_rc, r_dd))
            rec(cohort, vname, "spearman_DCS_RCS", r_dr, len(v))
            rec(cohort, vname, "spearman_RCS_conservation", r_rc, len(v))
            rec(cohort, vname, "spearman_DCS_dynamics", r_dd, len(v))

            lab = v.dropna(subset=["mean_am_pathogenicity"]).copy()
            lab = lab[(lab["mean_am_pathogenicity"] > AM_PATHOGENIC)
                      | (lab["mean_am_pathogenicity"] < AM_BENIGN)]
            pos = lab["mean_am_pathogenicity"] > AM_PATHOGENIC
            for score in ("rcs10", "dcs10"):
                A, npos, nneg = auc(lab[score], pos)
                log("    AUC %s %s   (%d positive, %d negative)"
                    % (score, "   none" if A is None else "%6.3f" % A,
                       npos, nneg))
                rec(cohort, vname, "auc_" + score, A, len(lab),
                    "%d pos %d neg" % (npos, nneg))
                cut = lab[score].quantile(1 - TOP_PCT)
                tail = lab[lab[score] >= cut]
                if len(tail):
                    prec = (tail["mean_am_pathogenicity"] > AM_PATHOGENIC).mean()
                    log("    top %.0f%% precision %s  %.3f  (n = %d)"
                        % (100 * TOP_PCT, score, prec, len(tail)))
                    rec(cohort, vname, "top1pct_precision_" + score, prec,
                        len(tail))

            cut = v["dcs10"].quantile(1 - TOP_DECILE)
            dec = v[v["dcs10"] >= cut]
            share = (dec["gap_frac"] > GAP_HI).mean()
            log("    top decile DCS: n = %d, unique residues %d, on gappy "
                "columns %.1f%%" % (len(dec), len(dec), 100 * share))
            rec(cohort, vname, "top_decile_DCS_n", len(dec), len(dec))
            rec(cohort, vname, "top_decile_DCS_gappy_share", share, len(dec))

            v["cohort"], v["variant"] = cohort, vname
            allscores.append(v[["cohort", "variant", "protein_id",
                                "UniProt_ID", "i", "A_i", "gap_frac",
                                "cons_reported", "cons_ungapped", "cons_used",
                                "nd", "dcs10", "rcs10",
                                "mean_am_pathogenicity"]])


        if cohort == "data-human":
            log("\n  the four discovery-zone variants of Table 1")
            log("    gene   variant       pos  gap    published DCS/RCS   "
                "recomputed D   survives gap filter")
            for gene, var, acc, posn, pdcs, prcs, _am in DISCOVERY:
                row = d[(d["UniProt_ID"] == acc) & (d["i"] == posn)]
                if not len(row):
                    log("    %-6s %-13s %4d  not matched to an alignment column"
                        % (gene, var, posn))
                    continue
                gf = float(row["gap_frac"].iloc[0])
                dv = allscores[-1]
                got = dv[(dv["UniProt_ID"] == acc) & (dv["i"] == posn)]
                surv = "yes" if gf <= GAP_HI else "NO, column is %.0f%% gap" % (
                    100 * gf)
                rd = ("%.2f/%.2f" % (float(got["dcs10"].iloc[0]),
                                     float(got["rcs10"].iloc[0]))
                      if len(got) else "dropped")
                log("    %-6s %-13s %4d  %.3f  %5.2f/%5.2f         %-12s   %s"
                    % (gene, var, posn, gf, pdcs, prcs, rd, surv))
                rec(cohort, "D_both", "discovery_%s_%s_gap" % (gene, var), gf,
                    1, surv)

    if len(allscores) < 4:
        raise Fatal("expected four variants per cohort, built %d frames"
                    % len(allscores))

    os.makedirs(a.outdir, exist_ok=True)
    sc = pd.concat(allscores, ignore_index=True)
    outp = os.path.join(a.outdir, "gap_survival_scores.csv.gz")
    with gzip.open(outp, "wt", newline="") as fh:
        sc.to_csv(fh, index=False)
    log("\nwrote %s (%d rows)" % (outp, len(sc)))
    outs = os.path.join(a.outdir, "gap_survival_summary.csv")
    pd.DataFrame(summary).to_csv(outs, index=False)
    log("wrote %s (%d rows)" % (outs, len(summary)))
    logp = os.path.join(a.outdir, "gap_survival_log.txt")
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        sys.exit("FATAL: %s" % exc)
