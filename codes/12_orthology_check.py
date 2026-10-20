#!/usr/bin/env python3

import csv
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(DATA, "ensembl_cache")
BASE = "https://rest.ensembl.org"
HUMAN_TAXON = "9606"


FAMILIES = [
    ("peroxidases", os.path.join(DATA, "an_peroxidase_PF03098.fasta")),
    ("syndecans", os.path.join(DATA, "families", "syndecans_all.fasta")),
    ("nectins", os.path.join(DATA, "families", "nectins_all.fasta")),
    ("JAMs", os.path.join(DATA, "families", "jams_all.fasta")),
    ("Eph receptors", os.path.join(DATA, "families", "ephs_all.fasta")),
    ("FGFRs", os.path.join(DATA, "families", "fgfrs_all_all.fasta")),
    ("connexins", os.path.join(DATA, "families", "gjbs_all_all.fasta")),
    ("Na channels", os.path.join(DATA, "families", "scns_all_all.fasta")),
]

MAX_TRIES = 8
BASE_WAIT = 4.0


class Fatal(RuntimeError):
    pass


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def get(path, key):

    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, key + ".json")
    if os.path.exists(cached):
        with open(cached) as fh:
            return json.load(fh)
    url = BASE + path
    wait = BASE_WAIT
    for attempt in range(1, MAX_TRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"Content-Type": "application/json",
                              "User-Agent": "conservation-score-reliability"})
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.load(r)
            with open(cached, "w") as fh:
                json.dump(payload, fh)
            return payload
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_TRIES:
                time.sleep(wait)
                wait = min(wait * 1.8, 90)
                continue
            raise Fatal("%s returned HTTP %d after %d attempts"
                        % (url, e.code, attempt))
        except Exception as e:
            if attempt < MAX_TRIES:
                time.sleep(wait)
                wait = min(wait * 1.8, 90)
                continue
            raise Fatal("%s failed after %d attempts: %s" % (url, attempt, e))
    raise Fatal("unreachable")


def species_taxa():

    d = get("/info/species?content-type=application/json", "_species")
    out = {}
    for s in d.get("species", []):
        if s.get("name") and s.get("taxon_id"):
            out[s["name"]] = str(s["taxon_id"])
    if len(out) < 100:
        raise Fatal("recovered %d Ensembl species with taxon ids, expected "
                    "hundreds; the join key is unusable" % len(out))
    return out


def entries(path):

    out = []
    for line in open(path):
        if not line.startswith(">"):
            continue
        acc = re.search(r"\|([^|]+)\|", line)
        gn = re.search(r"GN=(\S+)", line)
        ox = re.search(r"OX=(\d+)", line)
        name = re.search(r"\|([^|\s]+)\s", line)
        if not (gn and ox):
            continue
        out.append((acc.group(1) if acc else "?", gn.group(1).upper(),
                    ox.group(1), (name.group(1) if name else "?")))
    if not out:
        raise Fatal("no usable records parsed from %s" % path)
    return out


def main():
    os.makedirs(CACHE, exist_ok=True)
    logpath = os.path.join(DATA, "orthology_check_log.txt")
    with open(logpath, "w") as fh:
        log("12_orthology_check.py", fh)
        log("gene-name identity against Ensembl Compara one-to-one orthology\n", fh)

        taxa = species_taxa()
        log("Ensembl species with a taxon id: %d" % len(taxa), fh)

        rows, summary = [], []
        for fam, path in FAMILIES:
            if not os.path.exists(path):
                raise Fatal("%s is missing; family %s cannot be checked"
                            % (path, fam))
            recs = entries(path)
            human = sorted({g for _, g, t, _ in recs if t == HUMAN_TAXON})
            if not human:
                raise Fatal("%s holds no human sequence; there is no gene to "
                            "ask Compara about" % fam)
            log("\n%s: %d sequences, %d human genes" % (fam, len(recs), len(human)), fh)


            confirmed = {}
            for g in human:
                key = "orth_%s" % g
                d = get("/homology/symbol/human/%s?content-type=application/"
                        "json;type=orthologues;format=condensed" % g, key)
                data = d.get("data") or []
                hom = data[0].get("homologies", []) if data else []
                one2one, other = set(), set()
                for h in hom:
                    t = taxa.get(h.get("species", ""))
                    if not t:
                        continue
                    (one2one if h.get("type") == "ortholog_one2one"
                     else other).add(t)
                confirmed[g] = (one2one, other)
                log("  %-9s %4d Compara records, %3d one-to-one taxa"
                    % (g, len(hom), len(one2one)), fh)
                if not os.path.exists(os.path.join(CACHE, key + ".json")):
                    raise Fatal("%s was fetched but not cached" % g)
                time.sleep(1.0)

            n_ok = n_other = n_absent = n_nogene = 0
            for acc, gene, taxon, name in recs:
                if taxon == HUMAN_TAXON:
                    continue
                if gene not in confirmed:
                    verdict = "no human gene of that symbol in the family"
                    n_nogene += 1
                else:
                    one2one, other = confirmed[gene]
                    if taxon in one2one:
                        verdict = "one-to-one"
                        n_ok += 1
                    elif taxon in other:
                        verdict = "orthologue, not one-to-one"
                        n_other += 1
                    else:
                        verdict = "not in Compara"
                        n_absent += 1
                rows.append({"family": fam, "accession": acc, "entry": name,
                             "gene": gene, "taxon": taxon, "verdict": verdict})
            n = n_ok + n_other + n_absent + n_nogene
            if n == 0:
                raise Fatal("%s produced no non-human sequence to check" % fam)
            summary.append({"family": fam, "non_human_sequences": n,
                            "one_to_one": n_ok,
                            "orthologue_not_one_to_one": n_other,
                            "not_in_compara": n_absent,
                            "symbol_not_in_family": n_nogene,
                            "confirmed_fraction": round(n_ok / n, 4)})
            log("  %d non-human sequences: %d one-to-one (%.1f%%), %d other "
                "orthologue, %d not in Compara"
                % (n, n_ok, 100 * n_ok / n, n_other, n_absent), fh)

        a = os.path.join(DATA, "orthology_check_by_sequence.csv")
        with open(a, "w", newline="") as fh2:
            w = csv.DictWriter(fh2, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        b = os.path.join(DATA, "orthology_check_summary.csv")
        with open(b, "w", newline="") as fh2:
            w = csv.DictWriter(fh2, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)

        tot = sum(s["non_human_sequences"] for s in summary)
        ok = sum(s["one_to_one"] for s in summary)
        log("\npooled over %d families: %d of %d non-human sequences confirmed "
            "one-to-one, %.1f per cent" % (len(summary), ok, tot, 100 * ok / tot), fh)
        log("\nwrote %s (%d rows)" % (os.path.relpath(a, os.path.join(HERE, "..")), len(rows)), fh)
        log("wrote %s" % os.path.relpath(b, os.path.join(HERE, "..")), fh)


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        print("\nFATAL: %s" % exc, file=sys.stderr)
        sys.exit(1)
