"""
Generates (microstructure, kappa_eff) pairs for a SEED x TARGET-KAPPA sweep,
all at a SINGLE fixed regularization weight.                                   # <<< CHANGED

The grid is square and driven by ONE number, --num-samples N:
    - N seeds          : seed_start, seed_start + seed_step, ...  (N values)
    - N target-kappas  : kappa_start, kappa_start + kappa_step, ... (N values) # <<< CHANGED
    - total runs       : N x N   (e.g. N=10 -> 100 .nc files)

N is the single source of truth for both axes. The step sizes only control
the spacing between values, NOT how many there are -- so the total is always
exactly N x N and can never silently drift.

Reg-weight is held constant for every run (--reg-weight). Each target-kappa    # <<< CHANGED
value v is turned into an ISOTROPIC DIAGONAL tensor [v, 0, 0, v]; only the      # <<< CHANGED
diagonal moves. (For an xx-only anisotropic sweep, change build_target_kappa   # <<< CHANGED
below to [v, 0.0, 0.0, kyy_fixed].)                                             # <<< CHANGED

Each .nc filename encodes the grid resolution and both sweep coordinates, e.g.:
    result2D_g64_seed0003_k0.3000.nc                                           # <<< CHANGED

Everything is configurable from the command line:
    - N per axis                 (--num-samples ; gives N x N total)
    - seed start / step          (--seed-start --seed-step)
    - target-kappa start / step  (--kappa-start --kappa-step)                  # <<< CHANGED
    - explicit target-kappa list (--kappa-values ; overrides start/step, N long) # <<< CHANGED
    - the single fixed reg-weight (--reg-weight)                              # <<< CHANGED
    - output folder               (--output-dir)

Run `python seed_targetkappa_gen.py -h` for the full list.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------
# Tee: mirror console output into a log file too.
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


# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------

DEFAULT_NUM_SAMPLES = 10          # N per axis -> N x N total runs
DEFAULT_SEED_START = 1            # first seed value
DEFAULT_SEED_STEP = 1             # each next seed = prev + this

DEFAULT_KAPPA_START = 0.5         # first target-kappa (diagonal) value        # <<< CHANGED
DEFAULT_KAPPA_STEP = 0.05         # each next target-kappa = prev + this       # <<< CHANGED

DEFAULT_REG_WEIGHT = 0.062         # single fixed reg-weight for every run       # <<< ADDED

DEFAULT_MPI_PROCESSES = 4
DEFAULT_MAX_CONCURRENT_SIMS = 1
DEFAULT_GRID = 64

DEFAULT_BFGS_MAXITER = 500
DEFAULT_BFGS_GTOL = 2.5

SIMULATION_SCRIPT = "simulate_conduction.py"

KAPPA_SOLID = 1.0


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------

def validate_kappa(value):
    if not (0 < value < KAPPA_SOLID):
        raise ValueError(
            f"kappa target {value} is outside the achievable range "
            f"(0, {KAPPA_SOLID}) given kappa_solid={KAPPA_SOLID}."
        )


def build_linear_list(start, step, count):
    """Exactly `count` values: start, start+step, start+2*step, ...
    Used for BOTH axes so the grid is always count x count."""
    return [round(start + i * step, 6) for i in range(count)]


def build_target_kappa(v):                                                    # <<< ADDED
    """Turn a scalar diagonal value v into a full [xx, xy, yx, yy] tensor.
    Isotropic diagonal: xx = yy = v, off-diagonals 0. For an xx-only sweep,
    return [v, 0.0, 0.0, kyy_fixed] instead."""
    return [v, 0.0, 0.0, v]


def parse_converged(stdout_text):
    import re
    match = re.search(r"converged=(\d)", stdout_text)
    if match is None:
        return None
    return bool(int(match.group(1)))


def run_single_simulation(cfg, output_file, seed, kappa_value):               # <<< CHANGED
    target_kappa = build_target_kappa(kappa_value)   # per-run diagonal tensor # <<< CHANGED
    reg_weight = cfg.reg_weight                       # constant across sweep  # <<< CHANGED

    sim_command = [
        "python",
        SIMULATION_SCRIPT,
        "-n",
        str(cfg.grid_size[0]),
        str(cfg.grid_size[1]),
        "--target-kappa",
        *map(str, target_kappa),
        "--reg-weight",
        str(reg_weight),
        "--bfgs-maxiter",
        str(cfg.bfgs_maxiter),
        "--output",
        str(output_file),
        "--seed",
        str(seed),
    ]

    if cfg.use_mpi:
        sim_command = ["mpirun", "-n", str(cfg.mpi_processes), "--bind-to", "none"] + sim_command
    if cfg.use_gtol:
        sim_command += ["--bfgs-gtol", str(cfg.bfgs_gtol)]

    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    result = subprocess.run(sim_command, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"\n--- FAILED: {' '.join(sim_command)} ---")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, sim_command, output=result.stdout, stderr=result.stderr
        )

    converged = parse_converged(result.stdout)
    return converged, target_kappa, result.stdout


# ---------------------------------------------------------
# sweep body
# ---------------------------------------------------------

def generate_sweep(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "terminal_output.txt"
    with tee_to_file(log_path):
        _generate_sweep_body(cfg)


def _generate_sweep_body(cfg):
    N = cfg.num_samples

    # BOTH axes are exactly N long. N is the single source of truth. -------
    seeds = build_linear_list(cfg.seed_start, cfg.seed_step, N)
    seeds = [int(round(s)) for s in seeds]  # seeds are integers

    if cfg.kappa_values is not None:                                          # <<< CHANGED
        # explicit list still has to match N so the grid stays N x N
        if len(cfg.kappa_values) != N:                                       # <<< CHANGED
            raise SystemExit(
                f"--kappa-values has {len(cfg.kappa_values)} entries but "    # <<< CHANGED
                f"--num-samples is {N}. They must match so the grid stays "
                f"{N} x {N}."
            )
        kappas = [round(v, 6) for v in cfg.kappa_values]                     # <<< CHANGED
    else:
        kappas = build_linear_list(cfg.kappa_start, cfg.kappa_step, N)       # <<< CHANGED
    # ---------------------------------------------------------------------

    for v in kappas:                                                          # <<< CHANGED
        validate_kappa(v)                                                     # <<< CHANGED

    total_runs = len(seeds) * len(kappas)  # == N * N by construction         # <<< CHANGED

    print("\n" + "#" * 60)
    print(f"Seed x target-kappa sweep | reg_weight={cfg.reg_weight}")         # <<< CHANGED
    print(f"N (per axis)      = {N}")
    print(f"seeds ({len(seeds)})        = {seeds}")
    print(f"target_kappas ({len(kappas)}) = {kappas}  (as [v,0,0,v])")        # <<< CHANGED
    print(f"TOTAL RUNS        = {len(seeds)} x {len(kappas)} = {total_runs}") # <<< CHANGED
    print("#" * 60)

    metadata = []
    sweep_start = time.perf_counter()

    for seed in seeds:
        for kappa_value in kappas:                                           # <<< CHANGED
            # filename encodes resolution + both coordinates
            output_file = cfg.output_dir / (
                f"result2D_g{cfg.grid}_seed{seed:04d}_k{kappa_value:.4f}.nc"  # <<< CHANGED
            )

            print(f"\n[start] seed={seed} target_kappa={kappa_value:.4f} "    # <<< CHANGED
                  f"reg_weight={cfg.reg_weight} -> {output_file.name}")       # <<< CHANGED

            run_start = time.perf_counter()
            converged, target_kappa, stdout = run_single_simulation(
                cfg, output_file, seed, kappa_value                          # <<< CHANGED
            )
            run_time = time.perf_counter() - run_start

            if converged is False:
                print(f"WARNING: seed={seed} k={kappa_value:.4f} did NOT converge")  # <<< CHANGED
            elif converged is None:
                print(f"WARNING: could not parse convergence for seed={seed} k={kappa_value:.4f}")  # <<< CHANGED

            metadata.append({
                "seed": seed,
                "target_kappa_value": kappa_value,                          # <<< CHANGED
                "target_kappa": target_kappa,                               # <<< CHANGED (full tensor used)
                "reg_weight": cfg.reg_weight,                               # <<< CHANGED
                "output_file": str(output_file),
                "converged": converged,
                "runtime_seconds": f"{run_time:.2f}",
            })

            print(f"[done]  seed={seed} k={kappa_value:.4f} "                # <<< CHANGED
                  f"converged={converged} runtime={run_time:.2f}s")

    sweep_time = time.perf_counter() - sweep_start

    metadata_file = cfg.output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "reg_weight": cfg.reg_weight,                                    # <<< CHANGED
            "num_samples": N,
            "total_runs": total_runs,
            "seeds": seeds,
            "target_kappas": kappas,                                         # <<< CHANGED
            "seed_start": cfg.seed_start,
            "seed_step": cfg.seed_step,
            "kappa_start": cfg.kappa_start,                                  # <<< CHANGED
            "kappa_step": cfg.kappa_step,                                    # <<< CHANGED
            "kappa_values": cfg.kappa_values,                               # <<< CHANGED
            "grid_size": cfg.grid_size,
            "bfgs_maxiter": cfg.bfgs_maxiter,
            "use_mpi": cfg.use_mpi,
            "mpi_processes": cfg.mpi_processes,
            "sweep_runtime_seconds": f"{sweep_time:.2f}",
            "sweep_runtime_minutes": f"{sweep_time / 60:.2f}",
            "runs": metadata,
        }, f, indent=2)

    num_failed = sum(1 for m in metadata if m["converged"] is False)
    print(f"\nSweep done in {sweep_time:.2f}s ({sweep_time/60:.2f} min). "
          f"{num_failed}/{total_runs} runs did not converge.")
    print(f"Metadata written to {metadata_file}")


# ---------------------------------------------------------
# argument parsing
# ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Square N x N seed x target-kappa sweep at a single fixed reg-weight.",  # <<< CHANGED
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # THE grid size: one number drives both axes -> N x N total
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help="N per axis. Produces N seeds x N target-kappas = N*N total runs.")  # <<< CHANGED
    # seed axis (start + step only; count comes from --num-samples)
    p.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START,
                   help="First seed value.")
    p.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP,
                   help="Spacing between successive seeds.")
    # target-kappa axis (start + step only; count comes from --num-samples)   # <<< CHANGED
    p.add_argument("--kappa-start", type=float, default=DEFAULT_KAPPA_START,  # <<< CHANGED
                   help="First target-kappa diagonal value.")
    p.add_argument("--kappa-step", type=float, default=DEFAULT_KAPPA_STEP,    # <<< CHANGED
                   help="Spacing between successive target-kappa values.")
    p.add_argument("--kappa-values", type=float, nargs="+", default=None,     # <<< CHANGED
                   help="Explicit target-kappa list (must be exactly --num-samples long). "
                        "Overrides --kappa-start/--kappa-step; use for non-uniform spacing.")
    # fixed reg-weight                                                        # <<< CHANGED
    p.add_argument("--reg-weight", type=float, default=DEFAULT_REG_WEIGHT,    # <<< ADDED
                   help="Single fixed reg-weight applied to every run.")
    # sim config
    p.add_argument("--grid", type=int, default=DEFAULT_GRID,
                   help="Square grid size N; produces an N x N domain.")
    p.add_argument("--mpi-processes", type=int, default=DEFAULT_MPI_PROCESSES,
                   help="MPI ranks per simulation.")
    p.add_argument("--max-concurrent-sims", type=int, default=DEFAULT_MAX_CONCURRENT_SIMS,
                   help="(Reserved) simulations to run concurrently.")
    p.add_argument("--bfgs-maxiter", type=int, default=DEFAULT_BFGS_MAXITER,
                   help="L-BFGS max iterations.")
    p.add_argument("--bfgs-gtol", type=float, default=DEFAULT_BFGS_GTOL,
                   help="L-BFGS gradient tolerance.")
    p.add_argument("--no-mpi", action="store_true",
                   help="Disable mpirun; run simulate_conduction.py directly.")
    p.add_argument("--no-gtol", action="store_true",
                   help="Do not pass --bfgs-gtol to simulate_conduction.py.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory. If omitted, derived from reg-weight/grid.")  # <<< CHANGED

    args = p.parse_args()

    if args.num_samples < 1:
        p.error("--num-samples must be >= 1.")

    args.grid_size = [args.grid, args.grid]
    args.use_mpi = not args.no_mpi
    args.use_gtol = not args.no_gtol

    if args.output_dir is None:                                              # <<< CHANGED
        args.output_dir = Path("seed_targetkappa_sweep") / \
            f"rw{args.reg_weight}_g{args.grid}"
    else:
        args.output_dir = Path(args.output_dir)

    return args


def main():
    cfg = parse_args()
    print(f"Starting seed x target-kappa sweep at {time.ctime()}")           # <<< CHANGED
    print(f"grid={cfg.grid_size} reg_weight={cfg.reg_weight} "               # <<< CHANGED
          f"N={cfg.num_samples} -> {cfg.num_samples ** 2} runs out={cfg.output_dir}")
    overall_start = time.perf_counter()
    generate_sweep(cfg)
    overall_time = time.perf_counter() - overall_start
    print(f"\nComplete. Total time: {overall_time:.2f}s ({overall_time/60:.2f} min)")


if __name__ == "__main__":
    main()