#!/usr/bin/env python3

import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT = os.path.join(HERE, "..", "structures")
ENTRY = "7DE5"
URL = "https://files.rcsb.org/download/%s.pdb" % ENTRY
DEST = os.path.join(STRUCT, "%s.pdb" % ENTRY)

SHA256 = "f038e0ffd014f57f6ec979ee49878ae12a4add111524aa0fa5708c060791aea1"
CHAIN = "A"
CA_COUNT = 610
FIRST_RESIDUE = 1
LAST_RESIDUE = 595


class Fatal(Exception):
    pass


def chain_alpha_carbons(text):
    out = []
    for line in text.split("\n"):
        if (line.startswith("ATOM") and line[12:16].strip() == "CA"
                and line[21] == CHAIN):
            out.append(int(line[22:26]))
    return out


def check(raw):
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", "replace")
    ca = chain_alpha_carbons(text)
    if not ca:
        raise Fatal("no chain %s alpha carbons in the downloaded file" % CHAIN)
    if len(ca) != CA_COUNT:
        raise Fatal("chain %s has %d alpha carbons, expected %d; the entry has "
                    "been revised and the residue mapping in structures/ no "
                    "longer applies" % (CHAIN, len(ca), CA_COUNT))
    if ca[0] != FIRST_RESIDUE or ca[-1] != LAST_RESIDUE:
        raise Fatal("chain %s runs %d to %d, expected %d to %d"
                    % (CHAIN, ca[0], ca[-1], FIRST_RESIDUE, LAST_RESIDUE))
    if digest != SHA256:
        raise Fatal("sha256 is %s, expected %s; chain %s matches, so the file "
                    "differs outside the coordinates and the mapping still "
                    "applies, but the file is not byte-identical to the one "
                    "used here" % (digest, SHA256, CHAIN))
    return len(ca)


def main():
    if not os.path.isdir(STRUCT):
        os.makedirs(STRUCT)
    print("downloading %s" % URL)
    try:
        with urllib.request.urlopen(URL, timeout=120) as fh:
            raw = fh.read()
    except Exception as exc:
        raise Fatal("could not download %s: %s" % (URL, exc))
    if len(raw) < 10000:
        raise Fatal("downloaded %d bytes, which is too small to be the entry"
                    % len(raw))
    n = check(raw)
    with open(DEST, "wb") as fh:
        fh.write(raw)
    print("wrote %s" % DEST)
    print("  chain %s alpha carbons %d, residues %d to %d, sha256 matches"
          % (CHAIN, n, FIRST_RESIDUE, LAST_RESIDUE))


if __name__ == "__main__":
    try:
        main()
    except Fatal as exc:
        print("\nFATAL: %s" % exc, file=sys.stderr)
        sys.exit(1)
