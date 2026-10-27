"""
Off-diagonal (kxy) feasibility sweep -- kxx, kyy and kxy all swept independently.

Runs simulate_conduction.py on a regular grid of target tensors

        [[kxx, kxy],
         [kxy, kyy]]

    kxx : 0.15 -> 0.85, step 0.05   (15 values)
    kyy : 0.15 -> 0.85, step 0.05   (15 values, independent of kxx)   # <<< CHANGED
    kxy : 0.00 -> 0.50, step 0.05   (11 values, kyx = kxy)

-> 15 x 15 x 11 = 2475 runs (1644 pass the eigenvalue check).
   --half keeps only kyy >= kxx (1320 runs): the kxx<->kyy swap is the same
   structure rotated by 90 deg.                                        # <<< ADDED

Splitting across a SLURM array:                                         # <<< ADDED
    #SBATCH --array=0-9
    python kxy_sweep.py --task-id $SLURM_ARRAY_TASK_ID --num-tasks 10 ...
  Each task takes every num_tasks-th target, writes into the same folder
  and its own metadata_task<id>.json, so no two tasks collide.

Built from batch_parallelization_withtxt.py (thread pool, mpirun call, Tee
logging and convergence parsing unchanged). Every run uses the SAME seed, so
the target is the only thing that changes between runs. Infeasible targets
(eigenvalue outside (0, kappa_solid)) are still run on purpose and flagged
in the metadata.

Usage:
    python kxy_sweep.py --grid 64 --mpi-processes 2 --max-concurrent-sims 3 \
        --bfgs-gtol 1.0 --reg-weight 0.055
"""

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


# ---------------------------------------------------------
# Tee: mirror console output into a log file (unchanged)
# ---------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def tee_to_file(log_path):
    log_file = open(log_path, "w")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_out, log_file)
    sys.stderr = Tee(orig_err, log_file)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_file.close()


SIMULATION_SCRIPT = "simulate_conduction.py"
KAPPA_SOLID = 1.0


# ---------------------------------------------------------
# target grid + feasibility
# ---------------------------------------------------------

def frange(start, stop, step):
    """Inclusive float range rounded to 2 decimals (avoids 0.30000000004)."""
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 2) for i in range(n)]


def eigenvalues(kxx, kxy, kyy):
    return np.linalg.eigvalsh(np.array([[kxx, kxy], [kxy, kyy]]))


def is_feasible(kxx, kxy, kyy, kappa_solid=KAPPA_SOLID):
    """Necessary condition: both eigenvalues strictly inside (0, kappa_solid).
    Equivalent to kxx*kyy > kxy^2 and (1-kxx)(1-kyy) > kxy^2.
    (Real bounds are tighter: the larger eigenvalue cannot exceed the solid
    volume fraction -- Wiener bound -- so this only rules out the clearly
    impossible targets.)"""
    lam = eigenvalues(kxx, kxy, kyy)
    return bool(lam[0] > 1e-9 and lam[1] < kappa_solid - 1e-9)


def build_targets(cfg):
    jobs = []
    for kxx in frange(cfg.kxx_min, cfg.kxx_max, cfg.kxx_step):
        for kyy in frange(cfg.kyy_min, cfg.kyy_max, cfg.kyy_step):             # <<< CHANGED: independent kyy axis
            if cfg.half and kyy < kxx:                                          # <<< ADDED
                continue                                                        # <<< ADDED
            for kxy in frange(cfg.kxy_min, cfg.kxy_max, cfg.kxy_step):
                jobs.append((kxx, kxy, kyy))
    return jobs


# ---------------------------------------------------------
# single simulation
# ---------------------------------------------------------

def parse_converged(stdout_text):
    match = re.search(r"converged=(\d)", stdout_text)
    if match is None:
        return None
    return bool(int(match.group(1)))


def run_single_simulation(cfg, output_file, target_kappa):
    sim_command = [
        "python", SIMULATION_SCRIPT,
        "-n", str(cfg.grid), str(cfg.grid),
        "--target-kappa", *map(str, target_kappa),
        "--bfgs-maxiter", str(cfg.bfgs_maxiter),
        "--bfgs-gtol", str(cfg.bfgs_gtol),
        "--output", str(output_file),
        "--seed", str(cfg.seed),
    ]
    if cfg.reg_weight is not None:
        sim_command += ["--reg-weight", str(cfg.reg_weight)]
    if cfg.use_mpi:
        sim_command = ["mpirun", "-n", str(cfg.mpi_processes),
                       "--bind-to", "none"] + sim_command

    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1"})

    result = subprocess.run(sim_command, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"\n--- FAILED: {' '.join(sim_command)} ---")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        return None                       # one crash must not kill the sweep
    return parse_converged(result.stdout)


def out_name(out, kxx, kxy, kyy):                                               # <<< ADDED
    return out / f"result2D_kxx{kxx:.2f}_kyy{kyy:.2f}_kxy{kxy:.2f}.nc"         # <<< CHANGED: kyy in file name


# ---------------------------------------------------------
# sweep
# ---------------------------------------------------------

def run_sweep(cfg):
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    all_targets = build_targets(cfg)
    targets = all_targets[cfg.task_id::cfg.num_tasks]                          # <<< ADDED: this array task's share
    if cfg.skip_existing:                                                       # <<< ADDED: resume after timeout
        targets = [t for t in targets if not out_name(out, *t).exists()]        # <<< ADDED
    n_feas = sum(is_feasible(*t) for t in targets)
    print(f"task {cfg.task_id}/{cfg.num_tasks}: {len(targets)} of "
          f"{len(all_targets)} targets ({n_feas} feasible) | seed={cfg.seed} "
          f"| half={cfg.half}")

    metadata = []
    start = time.perf_counter()

    def worker(t):
        kxx, kxy, kyy = t
        target = [kxx, kxy, kxy, kyy]
        fname = out_name(out, kxx, kxy, kyy)
        lam = eigenvalues(kxx, kxy, kyy)
        feas = is_feasible(kxx, kxy, kyy)
        print(f"[start] kxx={kxx:.2f} kyy={kyy:.2f} kxy={kxy:.2f} "
              f"eig=({lam[0]:.2f},{lam[1]:.2f}) feasible={feas}")
        t0 = time.perf_counter()
        converged = run_single_simulation(cfg, fname, target)
        dt = time.perf_counter() - t0
        print(f"[done]  kxx={kxx:.2f} kyy={kyy:.2f} kxy={kxy:.2f} "
              f"converged={converged} {dt:.1f}s")
        return {
            "output_file": str(fname),
            "target_kxx": kxx, "target_kyy": kyy, "target_kxy": kxy,
            "target_eig_min": float(lam[0]), "target_eig_max": float(lam[1]),
            "feasible": feas,
            "converged": converged,
            "runtime_seconds": round(dt, 2),
        }

    with ThreadPoolExecutor(max_workers=cfg.max_concurrent_sims) as pool:
        futures = [pool.submit(worker, t) for t in targets]
        for f in as_completed(futures):
            metadata.append(f.result())

    metadata.sort(key=lambda m: (m["target_kxx"], m["target_kyy"], m["target_kxy"]))
    total = time.perf_counter() - start
    meta_path = out / f"metadata_task{cfg.task_id}.json"                       # <<< CHANGED: one file per array task
    with open(meta_path, "w") as f:
        json.dump({
            "grid": cfg.grid, "seed": cfg.seed, "half": cfg.half,
            "task_id": cfg.task_id, "num_tasks": cfg.num_tasks,
            "reg_weight": cfg.reg_weight, "bfgs_gtol": cfg.bfgs_gtol,
            "bfgs_maxiter": cfg.bfgs_maxiter,
            "sweep_runtime_minutes": round(total / 60, 2),
            "runs": metadata,
        }, f, indent=2)
    print(f"\nTask done in {total/60:.1f} min -> {meta_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--kxx-min", type=float, default=0.15)
    p.add_argument("--kxx-max", type=float, default=0.85)
    p.add_argument("--kxx-step", type=float, default=0.05)
    p.add_argument("--kyy-min", type=float, default=0.15)                       # <<< CHANGED
    p.add_argument("--kyy-max", type=float, default=0.85)                       # <<< CHANGED
    p.add_argument("--kyy-step", type=float, default=0.05)                      # <<< CHANGED
    p.add_argument("--kxy-min", type=float, default=0.0)
    p.add_argument("--kxy-max", type=float, default=0.5)
    p.add_argument("--kxy-step", type=float, default=0.05)
    p.add_argument("--half", action="store_true",                               # <<< ADDED
                   help="only run kyy >= kxx (the rest are 90-degree rotations)")
    p.add_argument("--task-id", type=int, default=0,                            # <<< ADDED
                   help="index of this SLURM array task")
    p.add_argument("--num-tasks", type=int, default=1,                          # <<< ADDED
                   help="total number of SLURM array tasks")
    p.add_argument("--skip-existing", action="store_true",                      # <<< ADDED
                   help="skip targets whose .nc already exists (resume)")
    p.add_argument("--seed", type=int, default=1000,
                   help="same initial density for every run")
    p.add_argument("--reg-weight", type=float, default=None,
                   help="passed to simulate_conduction.py (its default if omitted)")
    p.add_argument("--bfgs-maxiter", type=int, default=500)
    p.add_argument("--bfgs-gtol", type=float, default=2.5)
    p.add_argument("--mpi-processes", type=int, default=2)
    p.add_argument("--max-concurrent-sims", type=int, default=3)
    p.add_argument("--no-mpi", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    if not (0 <= args.task_id < args.num_tasks):
        raise SystemExit("--task-id must be in [0, --num-tasks)")
    args.use_mpi = not args.no_mpi
    if args.output_dir is None:
        args.output_dir = Path("kxy_sweep") / f"res_{args.grid}_seed{args.seed}"  # <<< CHANGED
    args.output_dir = Path(args.output_dir)

    total_cores_env = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("TOTAL_CORES")
    if args.use_mpi and total_cores_env:
        need = args.max_concurrent_sims * args.mpi_processes
        if need > int(total_cores_env):
            raise SystemExit(f"Oversubscription: {need} cores requested > {total_cores_env}")
    return args


def main():
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with tee_to_file(cfg.output_dir / f"terminal_output_task{cfg.task_id}.txt"):  # <<< CHANGED
        run_sweep(cfg)


if __name__ == "__main__":
    main()
