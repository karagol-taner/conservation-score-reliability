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
LOG2_20 = np.log2(20)
MIN_FILES = 100
BITSCORE = re.compile(r"_b(0\.\d)_")


class Fatal(RuntimeError):
    pass


def spearman(x, y):

    return x.rank().corr(y.rank())


def load_cohort(root, cohort, log):

    pat = os.path.join(root, cohort, "EVcouplings", "align_frequencies",
                       "*_frequencies.csv")
    files = sorted(glob.glob(pat))
    if not files:
        raise Fatal("no frequency files under %s" % pat)
    log("  %s: %d files" % (cohort, len(files)))
    frames = []
    for path in files:
        base = os.path.basename(path)
        df = pd.read_csv(path)
        missing = {"i", "A_i", "conservation", GAP} - set(df.columns)
        if missing:
            raise Fatal("%s lacks columns %s" % (base, sorted(missing)))
        aa_present = [c for c in AA if c in df.columns]
        if len(aa_present) != 20:
            raise Fatal("%s carries %d of 20 residue columns, so an ungapped "
                        "recomputation would be wrong" % (base, len(aa_present)))
        freq = df[aa_present].to_numpy(dtype=float)
        gap = df[GAP].to_numpy(dtype=float)
        mass = freq.sum(axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(mass[:, None] > 0, freq / mass[:, None], 0.0)
            ent = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)
        m = BITSCORE.search(base)
        frames.append(pd.DataFrame(dict(
            cohort=cohort, source=base,
            protein_id=base.split("_")[0],
            uniprot=base.split("_")[1] if "_" in base else "",
            bitscore=float(m.group(1)) if m else np.nan,
            i=df["i"].to_numpy(), A_i=df["A_i"].to_numpy(),
            conservation=df["conservation"].to_numpy(dtype=float),
            gap_frac=gap, ungapped_mass=mass,
            entropy_ungapped_bits=ent,
            cons_ungapped=1.0 - ent / LOG2_20)))
    out = pd.concat(frames, ignore_index=True)
    if out["gap_frac"].max() > 1.0001 or out["gap_frac"].min() < -1e-9:
        raise Fatal("gap fraction outside [0,1] in %s; the gap column is not a "
                    "frequency" % cohort)
    return out, len(files)


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

    log("03_gap_inflation_evcouplings.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("Does the reported conservation value rise with the gap fraction of its")
    log("own alignment column, and does the dynamics-aware result depend on it.")
    log("")
    log("gap threshold for a column being mostly absent evidence: %.2f" % GAP_HI)

    if not os.path.isdir(a.dyn):
        raise Fatal("no dynamics-aware folder at %s" % a.dyn)

    log("\nloading alignment columns")
    parts, nfiles = [], 0
    for cohort in ("data-human", "data-alphahelical"):
        df, n = load_cohort(a.dyn, cohort, log)
        parts.append(df)
        nfiles += n
    if nfiles < MIN_FILES:
        raise Fatal("only %d frequency files found, below the stated minimum "
                    "of %d; the cohort is incomplete" % (nfiles, MIN_FILES))
    pos = pd.concat(parts, ignore_index=True)
    log("  %d alignment columns over %d proteins in %d files"
        % (len(pos), pos["source"].nunique(), nfiles))

    summary = []

    def rec(measure, value, n=None, cohort="both", extra=""):
        summary.append(dict(measure=measure, cohort=cohort, n=n,
                            value=None if value is None else round(float(value), 4),
                            note=extra))


    log("\nassociation between gap fraction and conservation")
    log("  cohort            n       r(gap, cons)  rho(gap, cons)  "
        "r(gap, cons_ungapped)")
    for cohort in ("data-human", "data-alphahelical", "both"):
        d = pos if cohort == "both" else pos[pos["cohort"] == cohort]
        r1 = d["gap_frac"].corr(d["conservation"])
        s1 = spearman(d["gap_frac"], d["conservation"])
        r2 = d["gap_frac"].corr(d["cons_ungapped"])
        log("  %-16s %7d  %12.3f  %14.3f  %21.3f"
            % (cohort, len(d), r1, s1, r2))
        rec("pearson_gap_vs_conservation", r1, len(d), cohort)
        rec("spearman_gap_vs_conservation", s1, len(d), cohort)
        rec("pearson_gap_vs_conservation_ungapped", r2, len(d), cohort)

    hi = pos[pos["gap_frac"] > GAP_HI]
    log("\n  %d of %d columns (%.1f%%) are above %.0f%% gap, mean gap fraction "
        "%.3f overall" % (len(hi), len(pos), 100 * len(hi) / len(pos),
                          100 * GAP_HI, pos["gap_frac"].mean()))
    both_hi = pos[(pos["gap_frac"] > GAP_HI) & (pos["conservation"] > 0.8)]
    log("  %d columns (%.2f%%) are above %.0f%% gap AND report conservation "
        "above 0.80" % (len(both_hi), 100 * len(both_hi) / len(pos),
                        100 * GAP_HI))
    rec("columns_above_gap_threshold", len(hi) / len(pos), len(hi))
    rec("columns_gappy_and_high_conservation",
        len(both_hi) / len(pos), len(both_hi))

    log("\n  mean reported conservation by gap decile, which is the effect in "
        "one line")
    pos["gap_decile"] = (pos["gap_frac"] * 10).clip(0, 9).astype(int)
    g = pos.groupby("gap_decile").agg(
        n=("conservation", "size"), cons=("conservation", "mean"),
        cons_ungapped=("cons_ungapped", "mean"))
    for dec, row in g.iterrows():
        log("    gap %2d0-%2d0%%  n=%7d  conservation %.3f  ungapped %.3f"
            % (dec, dec + 1, row["n"], row["cons"], row["cons_ungapped"]))


    log("\nhow much of the dynamics-aware scored result sits on gappy columns")
    for cohort in ("data-human", "data-alphahelical"):
        master = os.path.join(a.dyn, cohort, "data_tsv",
                              "All_Proteins_DCS_Master.tsv")
        if not os.path.exists(master):
            log("  %s: no master table at %s" % (cohort, master))
            continue
        m = pd.read_csv(master, sep="\t")
        key = ["protein_id", "i"]
        sub = pos[pos["cohort"] == cohort][key + ["gap_frac", "conservation"]]
        m = m.rename(columns={"Protein_ID": "protein_id"})
        j = m.merge(sub, on=key, how="inner", suffixes=("", "_col"))
        log("  %s: %d of %d master residues matched to an alignment column"
            % (cohort, len(j), len(m)))
        if not len(j):
            continue
        base = (j["gap_frac"] > GAP_HI).mean()
        log("    base rate on gappy columns: %.1f%%" % (100 * base))
        rec("base_rate_gappy", base, len(j), cohort)
        for score in ("Dynamic_Conserved_Score_0_10",
                      "Rigid_Conserved_Score_0_10"):
            if score not in j.columns:
                continue
            s = j[score].astype(float)
            for frac, label in ((0.10, "top decile"), (0.01, "top 1%")):
                cut = s.quantile(1.0 - frac)
                tail = j[s >= cut]
                if not len(tail):
                    continue
                share = (tail["gap_frac"] > GAP_HI).mean()
                log("    %-32s %-11s n=%5d  on gappy columns %5.1f%%  "
                    "enrichment %.2fx"
                    % (score, label, len(tail), 100 * share,
                       share / base if base > 0 else float("nan")))
                rec("gappy_share_" + score + "_" + label.replace(" ", ""),
                    share, len(tail), cohort,
                    "enrichment %.2fx" % (share / base if base > 0 else -1))


    log("\ndoes alignment depth drive gappiness? the jackhmmer bitscore "
        "threshold")
    log("is in each filename and is not reported in the source publication")
    log("  bitscore  proteins  columns   mean gap  mean conservation")
    bs = pos.dropna(subset=["bitscore"])
    for b, grp in bs.groupby("bitscore"):
        log("  %8.1f  %8d  %7d  %9.3f  %17.3f"
            % (b, grp["source"].nunique(), len(grp), grp["gap_frac"].mean(),
               grp["conservation"].mean()))
        rec("mean_gap_at_bitscore_%.1f" % b, grp["gap_frac"].mean(), len(grp),
            "both", "%d proteins" % grp["source"].nunique())
    if bs["bitscore"].nunique() > 1:

        per = bs.groupby(["source", "bitscore"], as_index=False)["gap_frac"].mean()
        rho = spearman(per["bitscore"], per["gap_frac"])
        log("\n  Spearman(bitscore threshold, per-protein mean gap fraction) = "
            "%.3f over %d proteins" % (rho, len(per)))
        log("  a negative value means a more permissive threshold gives gappier "
            "columns,")
        log("  which would make alignment depth the single cause behind both "
            "failure modes")
        rec("spearman_bitscore_vs_gap", rho, len(per))


    os.makedirs(a.outdir, exist_ok=True)
    outp = os.path.join(a.outdir, "gap_inflation_positions.csv.gz")
    keep = ["cohort", "source", "protein_id", "uniprot", "bitscore", "i", "A_i",
            "conservation", "gap_frac", "ungapped_mass",
            "entropy_ungapped_bits", "cons_ungapped"]
    with gzip.open(outp, "wt", newline="") as fh:
        pos[keep].to_csv(fh, index=False)
    log("\nwrote %s (%d rows)" % (outp, len(pos)))

    outs = os.path.join(a.outdir, "gap_inflation_summary.csv")
    pd.DataFrame(summary).to_csv(outs, index=False)
    log("wrote %s (%d rows)" % (outs, len(summary)))

    logp = os.path.join(a.outdir, "gap_inflation_log.txt")
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        sys.exit("FATAL: %s" % exc)
