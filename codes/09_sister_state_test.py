#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


FAMILIES = [
    ("lpo",     "peroxidases",  "peroxidase", 4),
    ("sdc1",    "syndecans",    "syndecans",  4),
    ("nectin1", "nectins",      "nectins",    4),
    ("f11r",    "JAMs",         "jams",       3),
    ("epha2",   "Eph receptors", "ephs",     14),
    ("fgfr1",   "FGFRs",        "fgfrs",      4),
    ("gjb2",    "connexins",    "gjbs",       5),
    ("scn1a",   "Na channels",  "scns",      10),
]


CONSERVATIVE = {
    ("S", "A"), ("A", "S"), ("I", "L"), ("L", "I"), ("I", "V"), ("V", "I"),
    ("L", "V"), ("V", "L"), ("T", "S"), ("S", "T"), ("K", "R"), ("R", "K"),
    ("F", "L"), ("L", "F"), ("G", "A"), ("A", "G"), ("D", "E"), ("E", "D"),
    ("Q", "E"), ("E", "Q"), ("N", "D"), ("D", "N"), ("F", "Y"), ("Y", "F"),
    ("M", "L"), ("L", "M"), ("M", "I"), ("I", "M"), ("T", "A"), ("A", "T"),
    ("Q", "K"), ("K", "Q"), ("N", "S"), ("S", "N"), ("H", "Y"), ("Y", "H"),
    ("H", "R"), ("R", "H"),
}

REQUIRED = ("gene", "wt", "mut", "classification", "ref_position", "mapped",
            "verdict", "sister_state")
VOCAB = {"LB/B", "LP/P", "US"}


class Fatal(Exception):
    pass


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")


def fisher_one_sided(a, b, c, d):

    lf = lambda n: math.lgamma(n + 1)
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) == 0:
        return float("nan")

    def point(k):
        bb, cc, dd = (a + b) - k, (a + c) - k, n - (a + b) - (a + c) + k
        if min(bb, cc, dd) < 0:
            return 0.0
        return math.exp(lf(a + b) + lf(c + d) + lf(a + c) + lf(b + d) - lf(n)
                        - lf(k) - lf(bb) - lf(cc) - lf(dd))

    return sum(point(k) for k in range(a, min(a + b, a + c) + 1))


def woolf(a, b, c, d):

    if min(a, b, c, d) == 0:
        return float("nan"), float("nan")
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    lor = math.log((a * d) / (b * c))
    return math.exp(lor - 1.96 * se), math.exp(lor + 1.96 * se)


def mantel_haenszel(tables):

    num = den = obs = exp = var = 0.0
    used = 0
    Ps, Qs, Rs, Ss = [], [], [], []
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        used += 1
        num += a * d / n
        den += b * c / n
        obs += a
        exp += (a + b) * (a + c) / n
        var += ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1)) if n > 1 else 0.0
        Ps.append((a + d) / n); Qs.append((b + c) / n)
        Rs.append(a * d / n); Ss.append(b * c / n)
    if used == 0 or var <= 0:
        return float("nan"), (float("nan"), float("nan")), float("nan"), used
    or_mh = float("inf") if den == 0 else num / den
    z = (abs(obs - exp) - 0.5) / math.sqrt(var)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    if obs < exp:
        p = 1.0 - p
    Rt, St = sum(Rs), sum(Ss)
    if Rt == 0 or St == 0:
        return or_mh, (float("nan"), float("nan")), p, used
    rbg = (sum(pp * rr for pp, rr in zip(Ps, Rs)) / (2 * Rt * Rt)
           + sum(pp * ss + qq * rr
                 for pp, qq, rr, ss in zip(Ps, Qs, Rs, Ss)) / (2 * Rt * St)
           + sum(qq * ss for qq, ss in zip(Qs, Ss)) / (2 * St * St))
    se = math.sqrt(rbg)
    lor = math.log(or_mh)
    return or_mh, (math.exp(lor - 1.96 * se), math.exp(lor + 1.96 * se)), p, used


def load_variants(tag):
    path = os.path.join(DATA, "reliability_variants_%s.csv" % tag)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise Fatal("%s lacks required columns %s" % (path, missing))
    for r in rows:
        if r["classification"] not in VOCAB:
            raise Fatal("%s: classification %r outside the humsavar vocabulary"
                        % (path, r["classification"]))
    return rows


def tree_diagnostic_positions(stem):
    path = os.path.join(DATA, "tree_null_%s.csv" % stem)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return {int(r["position"]) for r in csv.DictReader(fh)
                if r["call_tree"] == "diagnostic"}


def table_for(rows):

    a = sum(1 for r in rows if r["sister_state"] == "True"  and r["classification"] == "LB/B")
    b = sum(1 for r in rows if r["sister_state"] == "True"  and r["classification"] == "LP/P")
    c = sum(1 for r in rows if r["sister_state"] == "False" and r["classification"] == "LB/B")
    d = sum(1 for r in rows if r["sister_state"] == "False" and r["classification"] == "LP/P")
    return a, b, c, d


def conservative(r):
    return (r["wt"], r["mut"]) in CONSERVATIVE


def run(restrict_to_tree, fh, records):
    arm = "tree-diagnostic positions only" if restrict_to_tree else "all mapped positions"
    log("\n" + "=" * 74, fh)
    log("ARM: %s" % arm, fh)
    log("=" * 74, fh)
    log("%-14s %6s %6s  %6s %6s   %7s %7s  %8s" %
        ("family", "sis_B", "sis_P", "oth_B", "oth_P", "sis_ben", "oth_ben", "p"), fh)

    strata, pooled = [], [0, 0, 0, 0]
    for tag, label, stem, nsub in FAMILIES:
        rows = load_variants(tag)
        if rows is None:
            log("  %-14s no variant table, skipped" % label, fh)
            continue
        rows = [r for r in rows if r["mapped"] == "True"
                and r["classification"] in ("LB/B", "LP/P")]
        if restrict_to_tree:
            keep = tree_diagnostic_positions(stem)
            if keep is None:
                raise Fatal("tree null for %s is missing, so the restricted arm "
                            "cannot be computed" % label)
            rows = [r for r in rows if r["ref_position"].isdigit()
                    and int(r["ref_position"]) in keep]
        if not rows:
            log("  %-14s no classified mapped variants" % label, fh)
            continue
        a, b, c, d = table_for(rows)
        for i, v in enumerate((a, b, c, d)):
            pooled[i] += v
        sb = a / (a + b) if a + b else float("nan")
        ob = c / (c + d) if c + d else float("nan")
        p = fisher_one_sided(a, b, c, d)
        log("%-14s %6d %6d  %6d %6d   %7s %7s  %8s" %
            (label, a, b, c, d,
             "%.3f" % sb if a + b else "-", "%.3f" % ob if c + d else "-",
             "%.3f" % p if p == p else "-"), fh)

        for sub in (True, False):
            part = [r for r in rows if conservative(r) is sub]
            if part:
                strata.append(table_for(part))
        records.append(dict(arm=arm, family=label, subfamilies=nsub,
                            sister_benign=a, sister_pathogenic=b,
                            other_benign=c, other_pathogenic=d))

    a, b, c, d = pooled
    log("%-14s %6d %6d  %6d %6d" % ("POOLED", a, b, c, d), fh)
    if a + b + c + d == 0:
        raise Fatal("no classified mapped variants in any family")
    if a + b == 0:
        raise Fatal("zero sister-state variants pooled across every family, "
                    "which means the sister-state join failed rather than that "
                    "the prediction is wrong")

    naive_or = (a * d) / (b * c) if b * c else float("inf")
    log("\n  unstratified, for comparison only:", fh)
    log("    odds ratio %.2f, one-sided Fisher p = %.4f"
        % (naive_or, fisher_one_sided(a, b, c, d)), fh)
    log("    this number is not the result: it compares families with benign", fh)
    log("    shares from %.2f to %.2f as though they were one population."
        % (min(x["other_benign"] / (x["other_benign"] + x["other_pathogenic"])
               for x in records if x["arm"] == arm
               and x["other_benign"] + x["other_pathogenic"]),
           max(x["other_benign"] / (x["other_benign"] + x["other_pathogenic"])
               for x in records if x["arm"] == arm
               and x["other_benign"] + x["other_pathogenic"])), fh)

    or_mh, (lo, hi), p_mh, used = mantel_haenszel(strata)
    log("\n  Mantel-Haenszel over %d of %d family x conservatism strata:"
        % (used, len(strata)), fh)
    log("    odds ratio %.2f, 95%% RBG interval %.2f to %.2f, one-sided p = %.4f"
        % (or_mh, lo, hi, p_mh), fh)
    log("    %d strata carried an empty margin and contributed nothing."
        % (len(strata) - used), fh)
    crude = (a * d) / (b * c) if b and c else float("nan")
    clo, chi = woolf(a, b, c, d)
    log("    crude, unstratified: odds ratio %.2f, 95%% interval %.2f to %.2f,"
        % (crude, clo, chi), fh)
    log("      which the stratified estimate is to be read against.", fh)
    if not (lo < or_mh < hi):
        raise Fatal("the Robins-Breslow-Greenland interval %.4f to %.4f does "
                    "not contain the estimate %.4f it was computed for"
                    % (lo, hi, or_mh))
    return or_mh, (lo, hi), p_mh, (a, b, c, d), (crude, clo, chi), used, len(strata)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=DATA)
    a = ap.parse_args()

    out = os.path.join(a.outdir, "sister_state_log.txt")
    with open(out, "w") as fh:
        log("09_sister_state_test.py", fh)
        log("Are variants to a sister subfamily's fixed state enriched for benign", fh)
        log("classification? Stratified on family and on substitution conservatism.", fh)

        records = []
        summary = []
        for restrict in (False, True):
            (or_mh, (lo, hi), p_mh, pooled, (crude, clo, chi),
             used, total) = run(restrict, fh, records)
            arm = "tree_diagnostic" if restrict else "all_mapped"
            summary.append(dict(arm=arm, or_mh=round(or_mh, 4),
                                ci_low=round(lo, 4), ci_high=round(hi, 4),
                                p_mh=round(p_mh, 5), strata_used=used,
                                strata_total=total,
                                or_crude=round(crude, 4),
                                crude_ci_low=round(clo, 4),
                                crude_ci_high=round(chi, 4),
                                sister_benign=pooled[0],
                                sister_pathogenic=pooled[1],
                                other_benign=pooled[2],
                                other_pathogenic=pooled[3]))

        log("\n" + "=" * 74, fh)
        log("The prediction was registered before these numbers existed and the", fh)
        log("direction is reported whichever way it came out.", fh)

    with open(os.path.join(a.outdir, "sister_state_summary.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(a.outdir, "sister_state_by_family.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print("\nwrote sister_state_summary.csv, sister_state_by_family.csv and "
          "sister_state_log.txt")


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        print("\nFATAL: %s" % exc, file=sys.stderr)
        sys.exit(1)
