"""Benchmark parallel enzymatic expansion across process counts.

Runs ``generate_network`` at a fixed generation count for several ``num_procs``
values, reporting wall-clock time and confirming the output size is identical
across process counts (parallel == serial).

Run from the repository root (so the in-tree ``doranet`` is used)::

    python bench/bench_parallel.py
    python bench/bench_parallel.py --starters CC(O)CO --gen 2 --procs 1,2,4,8

Note: each run writes a ``<job>__forward_saved_network.pgnet`` file via
``generate_network``; these are removed at the end.
"""

import argparse
import glob
import os
import sys
import time

# Prefer the in-tree doranet (repo root) over any installed copy, so the
# benchmark exercises local changes when run as ``python bench/...``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doranet.modules.enzymatic.generate_network import (  # noqa: E402
    generate_network,
)


def _run_once(starters, gen, max_atoms, num_procs):
    job = f"bench_np{num_procs}"
    start = time.time()
    network = generate_network(
        job_name=job,
        starters=starters,
        gen=gen,
        max_atoms=max_atoms,
        num_procs=num_procs,
    )
    elapsed = time.time() - start
    return (len(network.mols), len(network.rxns)), elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel enzymatic expansion."
    )
    parser.add_argument("--starters", default="CCO")
    parser.add_argument("--gen", type=int, default=2)
    parser.add_argument("--max-carbons", type=int, default=5)
    parser.add_argument(
        "--procs",
        default="1,2,4",
        help="comma-separated process counts (default: 1,2,4)",
    )
    args = parser.parse_args()

    procs = [int(p) for p in args.procs.split(",")]
    max_atoms = {"C": args.max_carbons}

    rows = []
    for num_procs in procs:
        sizes, elapsed = _run_once(
            args.starters, args.gen, max_atoms, num_procs
        )
        rows.append((num_procs, sizes, elapsed))

    baseline_sizes = rows[0][1]
    baseline_time = rows[0][2]
    print()
    print(f"starter={args.starters}  gen={args.gen}  max C={args.max_carbons}")
    print(f"{'np':>4} {'mols':>8} {'rxns':>8} {'time(s)':>9} {'speedup':>8}")
    identical = True
    for num_procs, sizes, elapsed in rows:
        speedup = baseline_time / elapsed if elapsed else float("nan")
        mismatch = sizes != baseline_sizes
        identical = identical and not mismatch
        flag = "  <-- size mismatch!" if mismatch else ""
        print(
            f"{num_procs:>4} {sizes[0]:>8} {sizes[1]:>8} "
            f"{elapsed:>9.1f} {speedup:>7.2f}x{flag}"
        )
    print()
    print(f"output identical across np: {identical}")

    for path in glob.glob("bench_np*__*_saved_network.pgnet"):
        os.remove(path)


if __name__ == "__main__":
    main()
