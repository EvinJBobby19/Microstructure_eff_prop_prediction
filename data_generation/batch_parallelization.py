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
"""

import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------
# Settings for the dataset generation
# ---------------------------------------------------------

NUM_SAMPLES_PER_BATCH = 1
NUM_BATCHES = 1          # how many independent, reproducible batches to generate
BASE_SEED = 1            # batch i uses seed = BASE_SEED + i

USE_MPI = True
MPI_PROCESSES = 4        # cores per simulation

# How many simulations to run concurrently. Set this to
MAX_CONCURRENT_SIMS = 1

GRID_SIZE = [64, 64]
SIMULATION_SCRIPT = "simulate_conduction.py"
BFGS_MAXITER = 200         # 200 was too low for targets near KAPPA_MAX; 500 fixed it in testing

USE_GTOL = True
BFGS_GTOL = 1e-06            # default is 1e-5; tighter tolerance for more accurate kappa_eff

OUTPUT_DIR = Path("batch_parallelization/res_128/testo")

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


def run_single_simulation(output_file, kappa_xx, kappa_yy):
    target_kappa = [kappa_xx, 0.0, 0.0, kappa_yy]

    sim_command = [
        "python",
        SIMULATION_SCRIPT,
        "-n",
        str(GRID_SIZE[0]),
        str(GRID_SIZE[1]),
        "--target-kappa",
        *map(str, target_kappa),
        "--bfgs-maxiter",
        str(BFGS_MAXITER),
        "--output",
        str(output_file),
        "--seed",
        str(BASE_SEED),
    ]

    if USE_MPI:
        sim_command = ["mpirun", "-n", str(MPI_PROCESSES)] + sim_command
    if USE_GTOL:
        sim_command += ["--bfgs-gtol", str(BFGS_GTOL)]

    # Wrap with /usr/bin/time -v to get a per-subprocess peak RSS that's safe
    # to read under thread concurrency.
    command = ["/usr/bin/time", "-v"] + sim_command

    # Thread-safe: each subprocess gets its own env dict, no shared state.
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    result = subprocess.run(command, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )

    converged = parse_converged(result.stdout)
    peak_memory_kb = parse_peak_memory_kb(result.stderr)

    return converged, target_kappa, result.stdout, peak_memory_kb


# ---------------------------------------------------------
# batch generation and main
# ---------------------------------------------------------

def generate_batch(batch_id):
    seed = BASE_SEED + batch_id
    random.seed(seed)

    batch_dir = OUTPUT_DIR / f"batch_{batch_id:02d}_seed{seed}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print(f"Batch {batch_id} | seed={seed} | {NUM_SAMPLES_PER_BATCH} samples "
          f"| {MAX_CONCURRENT_SIMS} concurrent sims x {MPI_PROCESSES} ranks each")
    print("#" * 60)

    # Pre-generate all kappa targets up front, single-threaded, so the RNG
    # sequence stays deterministic regardless of how threads get scheduled.
    jobs = []
    for run_id in range(NUM_SAMPLES_PER_BATCH):
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

        print(f"\n[start] Sample {run_id + 1}/{NUM_SAMPLES_PER_BATCH} "
              f"target kappa: xx={kappa_xx:.4f} yy={kappa_yy:.4f} -> {output_file}")

        run_start = time.perf_counter()
        converged, target_kappa, stdout, peak_memory_kb = run_single_simulation(
            output_file, kappa_xx, kappa_yy
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
        print(f"[done]  Sample {run_id + 1}/{NUM_SAMPLES_PER_BATCH} "
              f"converged={converged} runtime={run_time:.2f}s")
        return run_id, entry

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SIMS) as pool:
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
                "num_samples": NUM_SAMPLES_PER_BATCH,
                "grid_size": GRID_SIZE,
                "kappa_min": KAPPA_MIN,
                "kappa_max": KAPPA_MAX,
                "bfgs_maxiter": BFGS_MAXITER,
                "use_mpi": USE_MPI,
                "mpi_processes": MPI_PROCESSES,
                "max_concurrent_sims": MAX_CONCURRENT_SIMS,
                "batch_runtime_seconds": f"{batch_time:.2f}",
                "batch_runtime_minutes": f"{batch_time_minutes:.2f}",
                "runs": metadata,
            },
            f,
            indent=2,
        )

    num_failed = sum(1 for m in metadata if m["converged"] is False)
    print(f"\nBatch {batch_id} done in {batch_time:.2f}s. ({batch_time/60:.2f} minutes) "
          f"{num_failed}/{NUM_SAMPLES_PER_BATCH} runs did not converge.")
    print(f"Metadata written to {metadata_file}")

    return metadata_file


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Starting dataset generation at {time.ctime()}")
    overall_start = time.perf_counter()

    for batch_id in range(NUM_BATCHES):
        generate_batch(batch_id)

    overall_time = time.perf_counter() - overall_start
    print(f"\nAll batches completed for Grid Size {GRID_SIZE}. Total time: {overall_time:.2f} seconds. "
          f"({overall_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
