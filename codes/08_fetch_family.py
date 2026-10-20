#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone

API = "https://www.ebi.ac.uk/proteins/api/proteins"
MIN_PER_GENE = 3
MIN_TOTAL = 12
PAGE = 200
RETRIES = 3


class Fatal(RuntimeError):
    pass


def _page(gene, reviewed, offset):
    params = {"gene": gene, "size": str(PAGE), "offset": str(offset)}
    if reviewed:
        params["reviewed"] = "true"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as fh:
                return json.loads(fh.read().decode("utf-8")), url
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2 + 3 * attempt)
    raise Fatal("could not fetch %s offset %d after %d attempts: %s"
                % (gene, offset, RETRIES, last))


def fetch_gene(gene, tier, cap, log):

    out, urls = [], []
    rev, url = _page(gene, True, 0)
    urls.append(url)
    for e in rev:
        e["_reviewed"] = True
    out.extend(rev)
    if tier == "all":
        offset = 0
        while offset < cap:
            page, url = _page(gene, False, offset)
            urls.append(url)
            fresh = [e for e in page if not e.get("_reviewed")]
            for e in fresh:
                e["_reviewed"] = (e.get("info", {}) or {}).get("type") == "Swiss-Prot"
            out.extend(fresh)
            if len(page) < PAGE:
                break
            offset += PAGE
    log("  %d entries over %d request(s)" % (len(out), len(urls)))
    return out, urls[0]


def primary_genes(entry):
    out = []
    for g in entry.get("gene", []) or []:
        nm = g.get("name", {})
        if isinstance(nm, dict) and nm.get("value"):
            out.append(nm["value"])
    return out


def scientific_name(entry):
    for n in entry.get("organism", {}).get("names", []) or []:
        if n.get("type") == "scientific":
            return n.get("value", "?")
    return "?"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="family name, used in filenames")
    ap.add_argument("--genes", required=True, help="comma-separated gene symbols")
    ap.add_argument("--ref-gene", required=True,
                    help="gene whose human entry is the reference")
    ap.add_argument("--clade", default="Mammalia",
                    help="taxonomic name that must appear in the lineage")
    ap.add_argument("--tier", choices=("reviewed", "all"), default="reviewed")
    ap.add_argument("--cap", type=int, default=1000,
                    help="maximum unreviewed entries requested per gene")
    ap.add_argument("--max-per-gene", type=int, default=30,
                    help="cap kept sequences per subfamily, reviewed preferred, "
                         "so that subfamilies stay comparable in size")
    ap.add_argument("--len-tol", type=float, default=0.40,
                    help="drop sequences whose length differs from the human "
                         "reference by more than this fraction")
    ap.add_argument("--outdir", default=os.path.join("..", "data", "families"))
    a = ap.parse_args()

    genes = [g.strip() for g in a.genes.split(",") if g.strip()]
    if a.ref_gene not in genes:
        raise Fatal("--ref-gene %s is not in --genes" % a.ref_gene)

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("08_fetch_family.py  run "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log("family %s, %d genes, clade %s, tier %s"
        % (a.name, len(genes), a.clade, a.tier))
    log("orthology criterion: primary gene name identity within the clade, from")
    log("reviewed UniProt entries. No orthology database is used; see the script")
    log("docstring for what this criterion does and does not establish.")

    records = OrderedDict()
    per_gene = defaultdict(int)
    dropped = defaultdict(list)
    for gene in genes:
        entries, url = fetch_gene(gene, a.tier, a.cap, log)
        log("\n%s:" % gene)
        log("  %s" % url)
        seen_org = set()
        for e in entries:
            acc = e.get("accession")
            uid = e.get("id", acc)
            seq = (e.get("sequence", {}) or {}).get("sequence", "")
            org = scientific_name(e)
            lineage = e.get("organism", {}).get("lineage", []) or []
            prim = primary_genes(e)
            if not seq:
                dropped[gene].append((acc, org, "no sequence"))
                continue
            if a.clade not in lineage:
                dropped[gene].append((acc, org, "outside %s" % a.clade))
                continue
            if not any(p.upper() == gene.upper() for p in prim):
                dropped[gene].append((acc, org,
                                      "primary gene %s" % (prim or ["none"])[0]))
                continue
            key = (gene.upper(), org)
            if key in seen_org:
                dropped[gene].append((acc, org, "second entry for this organism"))
                continue
            if acc in records:
                raise Fatal("accession %s returned twice" % acc)
            seen_org.add(key)
            pname = (e.get("protein", {}).get("recommendedName", {})
                     .get("fullName", {}).get("value")
                     or e.get("protein", {}).get("submittedName", [{}])[0]
                     .get("fullName", {}).get("value", "")
                     or gene)
            tax = e.get("organism", {}).get("taxonomy", 0)
            records[acc] = dict(acc=acc, uid=uid, gene=gene, org=org, tax=tax,
                                pname=pname, seq=seq,
                                reviewed=bool(e.get("_reviewed")))
            per_gene[gene] += 1
        log("  kept %d in %s" % (per_gene[gene], a.clade))
        for acc, org, why in dropped[gene][:6]:
            log("    dropped %s %s: %s" % (acc, org[:34], why))
        if len(dropped[gene]) > 6:
            log("    and %d more, all listed in the CSV"
                % (len(dropped[gene]) - 6))


    ref = next((r for r in records.values()
                if r["gene"].upper() == a.ref_gene.upper()
                and r["org"] == "Homo sapiens"), None)
    if ref is None:
        raise Fatal("no human entry for the reference gene %s" % a.ref_gene)


    anchors = {}
    for g in genes:
        rs = [r for r in records.values() if r["gene"] == g]
        if not rs:
            continue
        human = next((r for r in rs if r["org"] == "Homo sapiens"), None)
        revd = next((r for r in rs if r["reviewed"]), None)
        if human is not None:
            anchors[g] = (len(human["seq"]), "human entry")
        elif revd is not None:
            anchors[g] = (len(revd["seq"]), "first reviewed entry")
        else:
            lens = sorted(len(r["seq"]) for r in rs)
            anchors[g] = (lens[len(lens) // 2], "median of the gene")
    too_odd = []
    for acc, r in list(records.items()):
        al = anchors.get(r["gene"], (None, ""))[0]
        if al and abs(len(r["seq"]) - al) / al > a.len_tol:
            too_odd.append(acc)
    for acc in too_odd:
        r = records.pop(acc)
        al = anchors[r["gene"]][0]
        dropped[r["gene"]].append((acc, r["org"], "length %d against %s anchor %d"
                                   % (len(r["seq"]), r["gene"], al)))
    log("\nlength filter at %.0f%% of each gene's own anchor: %d dropped"
        % (100 * a.len_tol, len(too_odd)))
    for g in genes:
        if g in anchors:
            log("  %-8s anchor %4d residues, from the %s"
                % (g, anchors[g][0], anchors[g][1]))


    by_gene = defaultdict(list)
    for r in records.values():
        by_gene[r["gene"]].append(r)
    kept = OrderedDict()
    for g in genes:
        rs = sorted(by_gene[g], key=lambda r: (not r["reviewed"], r["acc"]))
        for r in rs[:a.max_per_gene]:
            kept[r["acc"]] = r
        for r in rs[a.max_per_gene:]:
            dropped[g].append((r["acc"], r["org"], "over the per-gene cap"))
    records = kept
    per_gene = {g: sum(1 for r in records.values() if r["gene"] == g)
                for g in genes}
    if ref["acc"] not in records:
        raise Fatal("the reference entry was removed by filtering, which must "
                    "not happen")

    for g in genes:
        if per_gene[g] < MIN_PER_GENE:
            raise Fatal("gene %s kept only %d sequences, below the stated "
                        "minimum of %d; the subfamily cannot carry a "
                        "within-group entropy" % (g, per_gene[g], MIN_PER_GENE))
    if len(records) < MIN_TOTAL:
        raise Fatal("family total %d, below the stated minimum of %d"
                    % (len(records), MIN_TOTAL))

    log("\nfamily %s: %d sequences over %d genes" % (a.name, len(records),
                                                     len(genes)))
    for g in genes:
        nrev = sum(1 for r in records.values()
                   if r["gene"] == g and r["reviewed"])
        log("  %-8s %3d  (%d reviewed)" % (g, per_gene[g], nrev))
    log("  reference %s %s (%s)" % (ref["acc"], ref["uid"], ref["gene"]))

    os.makedirs(a.outdir, exist_ok=True)
    fa = os.path.join(a.outdir, "%s_%s.fasta" % (a.name, a.tier))
    with open(fa, "w") as fh:
        for r in records.values():
            fh.write(">sp|%s|%s %s %s OS=%s OX=%d GN=%s PE=1 SV=1\n"
                     % (r["acc"], r["uid"], r["gene"], r["pname"], r["org"],
                        r["tax"], r["gene"]))
            s = r["seq"]
            for i in range(0, len(s), 60):
                fh.write(s[i:i + 60] + "\n")
    log("\nwrote %s (%d sequences)" % (fa, len(records)))

    import csv
    dropped_csv = os.path.join(a.outdir, "%s_%s_excluded.csv" % (a.name, a.tier))
    with open(dropped_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "accession", "organism", "reason"])
        for g in genes:
            for acc, org, why in dropped[g]:
                w.writerow([g, acc, org, why])
    log("wrote %s (%d exclusions)" % (dropped_csv,
                                      sum(len(v) for v in dropped.values())))

    spec = ";".join("%s=%s " % (g, g) for g in genes)
    log("\nsubfamily argument for the panel and the tree null, trailing spaces "
        "matter:")
    log('  --subfamilies "%s"' % spec)
    log('  --ref %s --own %s' % (ref["uid"], a.ref_gene))

    logp = os.path.join(a.outdir, "%s_%s_fetch_log.txt" % (a.name, a.tier))
    with open(logp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", logp)


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        sys.exit("FATAL: %s" % exc)
