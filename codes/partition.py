#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_EXTRA_ROOTS = tuple(p for p in os.environ.get("CDI_EXTRA_ROOTS", "")
                     .split(os.pathsep) if p)


def sibling(*parts, must_exist=False):

    rel = os.path.join(*parts)
    here = _HERE
    seen = []
    while True:
        cand = os.path.join(here, rel)
        seen.append(cand)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    for root in _EXTRA_ROOTS:
        cand = os.path.join(root, rel)
        seen.append(cand)
        if os.path.exists(cand):
            return cand
    if must_exist:
        raise Fatal("could not find %r above %s; looked in:\n    %s"
                    % (rel, _HERE, "\n    ".join(seen[:6])))
    return os.path.join(_HERE, "..", rel)


H_MIN = 0.5
SEF_HI = 0.50
SEF_LO = 0.25
P_MAX = 0.05
N_PERM = 2000
MIN_SUBFAMILY_N = 2
MAX_GAP_FRAC = 0.25
INV_FAM_MAX = 0.50
INV_OWN_MIN = 0.90

GAP = "-"


class Fatal(RuntimeError):
    pass


def read_fasta(path):

    seqs, name = {}, None
    try:
        fh = open(path)
    except OSError as exc:
        raise Fatal("no such FASTA: %s (%s)" % (path, exc))
    with fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                name = line[1:]
                if name in seqs:
                    raise Fatal("duplicate FASTA header: %s" % name)
                seqs[name] = []
            elif name:
                seqs[name].append(line.strip())
    out = {k: "".join(v).upper() for k, v in seqs.items() if "".join(v)}
    if not out:
        raise Fatal("no sequences parsed from %s" % path)
    return out


def protein_name(header):
    m = re.search(r"\|\S+\s+(.+?)\s+OS=", header)
    return m.group(1) if m else header[:60]


def organism(header):
    m = re.search(r"OS=(.+?)\s+OX=", header)
    return m.group(1) if m else "?"


def align(seqs, guide_tree="upgma"):

    from pyfamsa import Aligner, Sequence
    objs = [Sequence(k.encode(), v.encode()) for k, v in seqs.items()]
    msa = Aligner(guide_tree=guide_tree).align(objs)
    return {s.id.decode(): s.sequence.decode() for s in msa}


def assign_labels(headers, patterns):

    labels, unlabelled = {}, []
    for h in headers:
        name = protein_name(h)
        hit = [lab for lab, pats in patterns.items() if any(p in name for p in pats)]
        if len(hit) > 1:
            raise Fatal("header matches subfamilies %s, so the partition is not "
                        "a partition: %s" % (sorted(hit), name))
        if hit:
            labels[h] = hit[0]
        else:
            unlabelled.append(h)
    return labels, unlabelled


def local_data(name, *sibling_parts):

    here = os.path.join(_HERE, "..", "data", name)
    if os.path.exists(here):
        return here
    return sibling(*sibling_parts) if sibling_parts else here


def _local_paths():

    p = os.path.join(_HERE, "local_paths.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise Fatal("codes/local_paths.json exists but cannot be read: %s" % exc)


def dynamics_dir():

    env = os.environ.get("CDI_DYNAMICS_DIR") or _local_paths().get("dynamics_dir")
    if env:
        return env
    raise Fatal("the dynamics-aware cohort is not redistributed here; set "
                "CDI_DYNAMICS_DIR to the folder holding its align_frequencies "
                "CSVs and scored master TSVs, or give \"dynamics_dir\" in "
                "codes/local_paths.json (see README)")


def humsavar_path():

    env = os.environ.get("CDI_HUMSAVAR") or _local_paths().get("humsavar")
    return env if env and os.path.exists(env) else None


def check_disjoint(ortho_headers, para_headers):

    both = set(ortho_headers) & set(para_headers)
    if both:
        raise Fatal("orthologue and paralogue sets overlap at %d sequence(s): %s"
                    % (len(both), sorted(both)[:3]))


def map_positions(msa, ref_header, ref_seq):

    if ref_header not in msa:
        raise Fatal("reference %s absent from the alignment" % ref_header)
    row = msa[ref_header]
    rebuilt = row.replace(GAP, "")
    if rebuilt != ref_seq:
        raise Fatal("the gapped reference row does not reconstruct its input "
                    "sequence (%d aligned residues against %d input); the "
                    "alignment is not trustworthy and no statistic was computed"
                    % (len(rebuilt), len(ref_seq)))
    out, pos = {}, 0
    for col, ch in enumerate(row):
        if ch != GAP:
            pos += 1
            out[pos] = col
    return out


def _entropy_bits(counter, n):
    if n <= 0:
        return 0.0
    h = -sum((v / n) * math.log2(v / n) for v in counter.values() if v)
    return abs(h)


def _sef(residues, labels):

    n = len(residues)
    if n == 0:
        return None, 0.0, 0.0
    h_total = _entropy_bits(Counter(residues), n)
    if h_total <= 0.0:
        return None, 0.0, 0.0
    h_within = 0.0
    for lab in set(labels):
        grp = [r for r, l in zip(residues, labels) if l == lab]
        h_within += (len(grp) / n) * _entropy_bits(Counter(grp), len(grp))
    mi = h_total - h_within
    return max(0.0, mi) / h_total, h_total, mi


def column_partition(residues, labels, rng, n_perm=N_PERM):

    sef, h_total, mi = _sef(residues, labels)
    if sef is None:
        return dict(n=len(residues), h_total=0.0, mi=0.0, sef=None,
                    sef_null=None, sef_adj=None, z=None, p_perm=None,
                    n_perm=0)
    res = np.array(residues)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _sef(list(rng.permutation(res)), labels)[0] or 0.0
    null_mean, null_sd = float(null.mean()), float(null.std(ddof=1))
    denom = 1.0 - null_mean
    sef_adj = max(0.0, (sef - null_mean) / denom) if denom > 1e-12 else 0.0
    z = (sef - null_mean) / null_sd if null_sd > 1e-12 else float("inf")
    p_perm = (1.0 + float((null >= sef - 1e-12).sum())) / (n_perm + 1.0)
    return dict(n=len(residues), h_total=h_total, mi=mi, sef=sef,
                sef_null=null_mean, sef_adj=sef_adj, z=z, p_perm=p_perm,
                n_perm=n_perm)


def classify(stat, h_min=H_MIN, sef_hi=SEF_HI, sef_lo=SEF_LO, p_max=P_MAX):

    if stat["sef"] is None or stat["h_total"] < h_min:
        return "conserved"
    if stat["sef_adj"] >= sef_hi and stat["p_perm"] <= p_max:
        return "diagnostic"
    if stat["sef_adj"] < sef_lo:
        return "tolerant"
    return "indeterminate"


def column_set(msa, col, ref_header, labels, min_subfamily_n=MIN_SUBFAMILY_N):

    raw = [(h, msa[h][col]) for h in msa if h != ref_header and h in labels]
    gapped = sum(1 for _, r in raw if r == GAP)
    gap_frac = gapped / len(raw) if raw else 1.0
    kept = [(h, r) for h, r in raw if r != GAP]
    sizes = Counter(labels[h] for h, _ in kept)
    small = {lab for lab, c in sizes.items() if c < min_subfamily_n}
    kept = [(h, r) for h, r in kept if labels[h] not in small]
    return ([r for _, r in kept], [labels[h] for h, _ in kept],
            gap_frac, sorted(small))


def identities(msa, col, ref_header, ref_residue, labels, own_label):

    fam = [msa[h][col] for h in msa
           if h != ref_header and h in labels and msa[h][col] != GAP]
    own = [msa[h][col] for h in msa
           if h != ref_header and labels.get(h) == own_label
           and msa[h][col] != GAP]
    f_fam = fam.count(ref_residue) / len(fam) if fam else None
    f_own = own.count(ref_residue) / len(own) if own else None
    return f_fam, len(fam), f_own, len(own)


def inverted(f_fam, f_own, fam_max=INV_FAM_MAX, own_min=INV_OWN_MIN):

    if f_fam is None or f_own is None:
        return None
    return bool(f_fam < fam_max and f_own >= own_min)


def distribution(residues):
    return " ".join("%s:%d" % (a, c) for a, c in Counter(residues).most_common())
