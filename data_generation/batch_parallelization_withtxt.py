"""
Generates (microstructure, kappa_eff) training pairs by calling
simulate_conduction.py with randomly sampled diagonal target-kappa tensors.

- Reproducible: every batch is seeded, and the seed is logged in metadata.json
  alongside every sample, so any batch can be regenerated exactly.
- Multiple batches: each batch uses a different seed,
  so we can generate more data later without ever repeating a batch.
- Input validation: kappa targets are checked against the physically
  achievable range before a run is attempted.
- Convergence tracking: stdout from simulate_conduction.py is parsed for
  "converged=0/1" and recorded per sample, so non-converged runs can be
  filtered out before CNN training instead of silently polluting the dataset.
- Concurrent: multiple simulations run at once via a thread pool, each
  blocking on its own `mpirun` subprocess (GIL is released during the wait,
  so this actually parallelizes across samples).

All run parameters (grid, mpi processes, concurrency, output dir, etc.) are
exposed as command-line arguments so the same script can be scheduled with
different values per SLURM array task. Run `python batch_parallelization.py -h`.
"""

import argparse
import contextlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------
# Tee: mirror everything printed to the console into a log file too.
# ---------------------------------------------------------

class Tee:
    """Write stream that forwards to several underlying streams at once."""

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
    """Within this block, everything printed to stdout/stderr is also
    appended to log_path. Restores the originals on exit."""
    log_file = open(log_path, "w")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_out, log_file)
    sys.stderr = Tee(orig_err, log_file)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_file.close()


# ---------------------------------------------------------
# Defaults (used when a flag is not passed on the command line)
# ---------------------------------------------------------

DEFAULT_NUM_SAMPLES_PER_BATCH = 10
DEFAULT_NUM_BATCHES = 100          # how many independent, reproducible batches
DEFAULT_BASE_SEED = 1000            # batch i uses seed = BASE_SEED + i

DEFAULT_MPI_PROCESSES = 2       # cores per simulation
DEFAULT_MAX_CONCURRENT_SIMS = 2  # how many simulations to run concurrently
DEFAULT_GRID = 64               # square grid; -> GRID_SIZE = [GRID, GRID]

DEFAULT_BFGS_MAXITER = 500       # 200 was too low near KAPPA_MAX; 500 fixed it
DEFAULT_BFGS_GTOL = 2.5        # simulate_conduction.py default is 2.5

SIMULATION_SCRIPT = "simulate_conduction.py"

# Kept below kappa_solid=1.0: targets close to 1.0 push toward a degenerate
# all-solid structure and are more likely to fail to converge.
KAPPA_SOLID = 1.0
KAPPA_MIN = 0.05
KAPPA_MAX = 0.85


# ---------------------------------------------------------
# functions and other code for generating the dataset
# ---------------------------------------------------------

def validate_kappa(value):
    """Reject targets that are outside the physically achievable range."""
    if not (0 < value < KAPPA_SOLID):
        raise ValueError(
            f"kappa target {value} is outside the achievable range "
            f"(0, {KAPPA_SOLID}) given kappa_solid={KAPPA_SOLID}."
        )


def parse_converged(stdout_text):
    """Pull the converged=0/1 flag out of simulate_conduction.py's stdout."""
    match = re.search(r"converged=(\d)", stdout_text)
    if match is None:
        return None  # could not determine; flag for manual review
    return bool(int(match.group(1)))


def parse_peak_memory_kb(time_stderr_text):
    """
    Parse 'Maximum resident set size (kbytes): N' out of `/usr/bin/time -v` output.
    This is scoped to the single subprocess tree it wraps, so it stays correct
    even when multiple sims run concurrently (unlike resource.getrusage(RUSAGE_CHILDREN),
    which aggregates across the whole Python process and breaks under concurrency).
    """
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_stderr_text)
    if match is None:
        return None
    return int(match.group(1))


def run_single_simulation(cfg, output_file, kappa_xx, kappa_yy):
    target_kappa = [kappa_xx, 0.0, 0.0, kappa_yy]

    sim_command = [
        "python",
        SIMULATION_SCRIPT,
        "-n",
        str(cfg.grid_size[0]),
        str(cfg.grid_size[1]),
        "--target-kappa",
        *map(str, target_kappa),
        "--bfgs-maxiter",
        str(cfg.bfgs_maxiter),
        "--output",
        str(output_file),
        "--seed",
        str(cfg.base_seed),
    ]

    if cfg.use_mpi:
        sim_command = ["mpirun", "-n", str(cfg.mpi_processes),"--bind-to","none"] + sim_command
        # sim_command = ["srun","--exclusive","--exact","--cpu-bind=cores",
        #                "--ntasks",str(cfg.mpi_processes)] + sim_command
    if cfg.use_gtol:
        sim_command += ["--bfgs-gtol", str(cfg.bfgs_gtol)]

    # Wrap with /usr/bin/time -v to get a per-subprocess peak RSS that's safe
    # to read under thread concurrency.
    command = sim_command  # removed: command = ["/usr/bin/time", "-v"] + sim_command

    # Thread-safe: each subprocess gets its own env dict, no shared state.
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    result = subprocess.run(command, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"\n--- FAILED: {' '.join(command)} ---")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )

    converged = parse_converged(result.stdout)
    peak_memory_kb = 1024  # parse_peak_memory_kb(result.stderr)

    return converged, target_kappa, result.stdout, peak_memory_kb


# ---------------------------------------------------------
# batch generation and main
# ---------------------------------------------------------

def generate_batch(cfg, batch_id):
    seed = cfg.base_seed + batch_id
    random.seed(seed)

    batch_dir = cfg.output_dir / f"batch_{batch_id:02d}_seed{seed}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Mirror all console output for this batch into a .txt in the batch folder.
    log_path = batch_dir / f"terminal_output_seed{seed}.txt"
    with tee_to_file(log_path):
        metadata_file = _generate_batch_body(cfg, batch_id, seed, batch_dir)
    return metadata_file


def _generate_batch_body(cfg, batch_id, seed, batch_dir):
    print("\n" + "#" * 60)
    print(f"Batch {batch_id} | seed={seed} | {cfg.num_samples} samples "
          f"| {cfg.max_concurrent_sims} concurrent sims x {cfg.mpi_processes} ranks each")
    print("#" * 60)

    # Pre-generate all kappa targets up front, single-threaded, so the RNG
    # sequence stays deterministic regardless of how threads get scheduled.
    jobs = []
    for run_id in range(cfg.num_samples):
        kappa_xx = random.uniform(KAPPA_MIN, KAPPA_MAX)
        kappa_yy = random.uniform(KAPPA_MIN, KAPPA_MAX)

        validate_kappa(kappa_xx)
        validate_kappa(kappa_yy)

        output_file = batch_dir / f"result2D_{run_id:03d}.nc"
        jobs.append((run_id, output_file, kappa_xx, kappa_yy))

    metadata = [None] * len(jobs)
    batch_start = time.perf_counter()

    def worker(job):
        run_id, output_file, kappa_xx, kappa_yy = job

        print(f"\n[start] Sample {run_id + 1}/{cfg.num_samples} "
              f"target kappa: xx={kappa_xx:.4f} yy={kappa_yy:.4f} -> {output_file}")

        run_start = time.perf_counter()
        converged, target_kappa, stdout, peak_memory_kb = run_single_simulation(
            cfg, output_file, kappa_xx, kappa_yy
        )
        run_time = time.perf_counter() - run_start

        if converged is False:
            print(f"WARNING: run {run_id} did NOT converge (converged=0)")
        elif converged is None:
            print(f"WARNING: could not parse convergence status for run {run_id}")

        entry = {
            "run_id": run_id,
            "output_file": str(output_file),
            "target_kappa_xx": kappa_xx,
            "target_kappa_yy": kappa_yy,
            "converged": converged,
            "runtime_seconds": f"{run_time:.2f}",
            "peak_memory_mb": (
                f"{peak_memory_kb / 1024.0:.2f}" if peak_memory_kb is not None else None
            ),
        }
        print(f"[done]  Sample {run_id + 1}/{cfg.num_samples} "
              f"converged={converged} runtime={run_time:.2f}s")
        return run_id, entry

    with ThreadPoolExecutor(max_workers=cfg.max_concurrent_sims) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            run_id, entry = future.result()
            metadata[run_id] = entry

    batch_time = time.perf_counter() - batch_start
    batch_time_minutes = batch_time / 60.0

    metadata_file = batch_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "batch_id": batch_id,
                "seed": seed,
                "num_samples": cfg.num_samples,
                "grid_size": cfg.grid_size,
                "kappa_min": KAPPA_MIN,
                "kappa_max": KAPPA_MAX,
                "bfgs_maxiter": cfg.bfgs_maxiter,
                "use_mpi": cfg.use_mpi,
                "mpi_processes": cfg.mpi_processes,
                "max_concurrent_sims": cfg.max_concurrent_sims,
                "batch_runtime_seconds": f"{batch_time:.2f}",
                "batch_runtime_minutes": f"{batch_time_minutes:.2f}",
                "runs": metadata,
            },
            f,
            indent=2,
        )

    num_failed = sum(1 for m in metadata if m["converged"] is False)
    print(f"\nBatch {batch_id} done in {batch_time:.2f}s. ({batch_time/60:.2f} minutes) "
          f"{num_failed}/{cfg.num_samples} runs did not converge.")
    print(f"Metadata written to {metadata_file}")

    return metadata_file


# ---------------------------------------------------------
# argument parsing
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate (microstructure, kappa_eff) training pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--grid", type=int, default=DEFAULT_GRID,
                   help="Square grid size N; produces an N x N domain.")
    p.add_argument("--mpi-processes", type=int, default=DEFAULT_MPI_PROCESSES,
                   help="MPI ranks per simulation.")
    p.add_argument("--max-concurrent-sims", type=int, default=DEFAULT_MAX_CONCURRENT_SIMS,
                   help="Number of simulations to run concurrently (thread pool size).")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES_PER_BATCH,
                   help="Samples generated per batch.")
    p.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES,
                   help="Independent, reproducible batches to generate.")
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                   help="Batch i uses seed = base-seed + i; also pins the density field.")
    p.add_argument("--bfgs-maxiter", type=int, default=DEFAULT_BFGS_MAXITER,
                   help="L-BFGS max iterations.")
    p.add_argument("--bfgs-gtol", type=float, default=DEFAULT_BFGS_GTOL,
                   help="L-BFGS gradient tolerance.")
    p.add_argument("--no-mpi", action="store_true",
                   help="Disable mpirun; run simulate_conduction.py directly.")
    p.add_argument("--no-gtol", action="store_true",
                   help="Do not pass --bfgs-gtol to simulate_conduction.py.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory. If omitted, a name is derived from "
                        "grid/mpi/concurrency so parallel array tasks don't collide.")

    args = p.parse_args()

    # Derived / normalized fields, collected on a simple config object.
    args.grid_size = [args.grid, args.grid]
    args.use_mpi = not args.no_mpi
    args.use_gtol = not args.no_gtol

    if args.output_dir is None:
        # e.g. batch_parallelization/res_128/g128_mpi20_c1
        args.output_dir = Path("batch_parallelization") / f"res_{args.grid}" / \
            f"g{args.grid}_mpi{args.mpi_processes}_c{args.max_concurrent_sims}"
    else:
        args.output_dir = Path(args.output_dir)

    # Oversubscription guard: total cores requested must fit the allocation.
    # (Onl]y meaningful when using MPI; adjust TOTAL_CORES via SLURM env if set.)
    total_requested = args.max_concurrent_sims * args.mpi_processes
    total_cores_env = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get("TOTAL_CORES")
    if args.use_mpi and total_cores_env:
        total_cores = int(total_cores_env)
        if total_requested > total_cores:
            raise SystemExit(
                f"Oversubscription: max_concurrent_sims ({args.max_concurrent_sims}) "
                f"x mpi_processes ({args.mpi_processes}) = {total_requested} "
                f"> available cores ({total_cores})."
            )

    return args


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting dataset generation at {time.ctime()}")
    print(f"grid={cfg.grid_size} mpi={cfg.mpi_processes} "
          f"concurrent={cfg.max_concurrent_sims} out={cfg.output_dir}")
    overall_start = time.perf_counter()

    for batch_id in range(cfg.num_batches):
        generate_batch(cfg, batch_id)

    overall_time = time.perf_counter() - overall_start
    print(f"\nAll batches completed for Grid Size {cfg.grid_size}. "
          f"Total time: {overall_time:.2f} seconds. ({overall_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
