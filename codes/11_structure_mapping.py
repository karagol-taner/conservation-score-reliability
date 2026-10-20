#!/usr/bin/env python3

import csv
import json
import os
import sys

import partition as P

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(HERE, "..", "structures")
DATA = os.path.join(HERE, "..", "data")
PDB = os.path.join(STRUCT, "7DE5.pdb")
MAPFILE = os.path.join(STRUCT, "7DE5_to_human.json")
ALIGN = os.path.join(DATA, "an_peroxidase_PF03098.fasta")
REF = "PERL_HUMAN"
OUT = os.path.join(DATA, "structure_mapping_7DE5.csv")
LOG = os.path.join(DATA, "structure_mapping_log.txt")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}


CONTROL = (225, 226, 227, 301, 303, 305, 307, 372, 375, 468)

QUOTED = {144: "D", 205: "E", 313: "E", 648: "E", 479: "D", 169: "E", 700: "D"}


class Fatal(RuntimeError):
    pass


def log(msg, fh):
    print(msg)
    fh.write(msg + "\n")


def chain_a_residues():

    out = {}
    for line in open(PDB):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[21] != "A" or line[12:16].strip() != "CA":
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        three = line[17:20].strip().upper()
        if three not in THREE_TO_ONE:
            continue
        num = int(line[22:26])
        out.setdefault(num, THREE_TO_ONE[three])
    if not out:
        raise Fatal("no chain A alpha carbons parsed from %s" % PDB)
    return out


def human_sequence():

    seq, grabbing = [], False
    for line in open(ALIGN):
        if line.startswith(">"):
            grabbing = REF in line
            continue
        if grabbing:
            seq.append(line.strip())
    s = "".join(seq).replace("-", "").replace(".", "").upper()
    if not s:
        raise Fatal("%s not found in %s" % (REF, ALIGN))
    return s


def main():
    with open(LOG, "w") as fh:
        log("11_structure_mapping.py", fh)
        log("PDB 7DE5 chain A against human lactoperoxidase (%s)\n" % REF, fh)

        pdb = chain_a_residues()
        human = human_sequence()
        deposited = {int(k): int(v) for k, v in json.load(open(MAPFILE)).items()}
        log("  chain A alpha carbons   %d" % len(pdb), fh)
        log("  human sequence length   %d" % len(human), fh)
        log("  deposited map entries   %d" % len(deposited), fh)

        rows, agree = [], 0
        for pdb_num in sorted(deposited):
            hum_num = deposited[pdb_num]
            if pdb_num not in pdb:
                raise Fatal("map sends PDB residue %d to human %d, but %d has "
                            "no alpha carbon in chain A" % (pdb_num, hum_num,
                                                            pdb_num))
            if not (1 <= hum_num <= len(human)):
                raise Fatal("map sends PDB residue %d to human position %d, "
                            "outside the %d-residue sequence"
                            % (pdb_num, hum_num, len(human)))
            a, b = pdb[pdb_num], human[hum_num - 1]
            agree += a == b
            rows.append({"pdb_residue": pdb_num, "pdb_aa": a,
                         "human_position": hum_num, "human_aa": b,
                         "identical": "yes" if a == b else "no"})

        ident = agree / len(rows)
        log("\n  %d residues map" % len(rows), fh)
        log("  %d identical, %.1f per cent" % (agree, 100 * ident), fh)

        if len(rows) != 595:
            raise Fatal("the Methods state 595 mapped residues, the map gives "
                        "%d" % len(rows))
        if abs(round(100 * ident) - 85) > 0.5:
            raise Fatal("the Methods state 85 per cent identity, the map gives "
                        "%.1f per cent" % (100 * ident))


        log("\n  positions quoted in the text", fh)
        by_human = {r["human_position"]: r for r in rows}
        missing = []
        for pos in sorted(set(CONTROL) | set(QUOTED)):
            expect = QUOTED.get(pos)
            r = by_human.get(pos)
            if r is None:
                missing.append(pos)
                log("    %-5d not covered by the mapped chain" % pos, fh)
                continue
            if expect is not None and r["human_aa"] != expect:
                raise Fatal("human position %d should be %s, the sequence has "
                            "%s" % (pos, expect, r["human_aa"]))
            if pos in CONTROL and r["identical"] != "yes":
                raise Fatal("control position %d is %s in the sequence and %s in "
                            "the structure; the Methods state the ten control "
                            "positions carry their expected residue"
                            % (pos, r["human_aa"], r["pdb_aa"]))
            log("    %-5d %s in the sequence, %s in the structure, %s"
                % (pos, r["human_aa"], r["pdb_aa"],
                   "matches" if r["identical"] == "yes" else "differs"), fh)
        if missing:
            log("\n  %d quoted positions fall outside the deposited chain"
                % len(missing), fh)

        with open(OUT, "w", newline="") as out:
            w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        root = os.path.join(HERE, "..")
        log("\nwrote %s (%d rows)"
            % (os.path.relpath(OUT, root).replace(os.sep, "/"), len(rows)), fh)
        log("wrote %s" % os.path.relpath(LOG, root).replace(os.sep, "/"), fh)


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        print("\nFATAL: %s" % exc, file=sys.stderr)
        sys.exit(1)
