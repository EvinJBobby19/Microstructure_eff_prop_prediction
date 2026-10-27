#!/usr/bin/env python3
"""
Generate Cahn-Hilliard (spinodal-decomposition) microstructures as an
out-of-distribution test set for the kappa_eff CNN surrogate.

Pipeline per sample
-------------------
1. Solve the Cahn-Hilliard equation on a periodic grid (spectral,
   semi-implicit, stabilized):
       dc/dt = lap( c^3 - c - eps^2 * lap(c) ),   c in [-1, 1]
   Mean composition c0 controls the volume fraction (vf ~ (1 + c0) / 2),
   final time t_end controls the coarsening length scale, and an optional
   anisotropic gradient energy (eps_x != eps_y) gives elongated domains
   (kxx != kyy).
2. Threshold c >= 0 -> rho = 1 (solid), else 0 (void).
3. Homogenize rho with muTopOpt exactly like simulate_conduction.py
   (same SimpConductivity / HomogenizationConductivity settings), giving
   kappa_thresh = homog.homogenized_tangent().
4. Write a .nc file in the same layout as simulate_conduction.py
   ("density" field, global attributes kappa_eff / kappa_thresh /
   homog_data / converged / nb_grid_pts / command_line) plus a per-batch
   metadata.json, so the existing CNN data loader can read it unchanged.

Examples
--------
    # 200 test structures on 64x64, mixed isotropic/anisotropic
    python generate_cahn_hilliard.py -n 64 64 --num-samples 200 --seed 1000

    # only isotropic CH, preview PNG of the first 16
    python generate_cahn_hilliard.py -n 64 64 --num-samples 50 --aniso-prob 0 --preview

    general format of cahn hillard generation
python generate_cahn_hilliard.py -n 64 64 --num-samples 200 --seed 1000 \
    --void-ratio <same value batch_parallelization used> --preview
"""

import argparse
import json
import os
import shlex
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
# Cahn-Hilliard solver (pure numpy, no muGrid needed)
# --------------------------------------------------------------------------- #
def cahn_hilliard(nb_grid_pts, c0, eps_x, eps_y, t_end, dt, rng,
                  noise=0.05, stab=2.0):
    """Periodic 2D Cahn-Hilliard, stabilized semi-implicit Fourier scheme.

    Lengths are in grid spacings (h = 1). eps ~ interface width in pixels.
    Returns the order parameter c (shape nb_grid_pts), values ~ [-1, 1].
    """
    nx, ny = nb_grid_pts
    c = c0 + noise * rng.standard_normal((nx, ny))
    c += c0 - c.mean()                      # enforce exact mean composition

    kx = 2.0 * np.pi * np.fft.fftfreq(nx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k4_aniso = k2 * (eps_x ** 2 * KX ** 2 + eps_y ** 2 * KY ** 2)

    denom = 1.0 + dt * stab * k2 + dt * k4_aniso
    c_hat = np.fft.rfft2(c)
    nsteps = int(np.ceil(t_end / dt))
    for _ in range(nsteps):
        nonlin_hat = np.fft.rfft2(c ** 3 - c)
        c_hat = (c_hat * (1.0 + dt * stab * k2) - dt * k2 * nonlin_hat) / denom
        c = np.fft.irfft2(c_hat, s=(nx, ny))
    return c


def sample_params(rng, args):
    """Random CH parameters for one structure."""
    c0 = rng.uniform(args.c0_min, args.c0_max)
    eps = rng.uniform(args.eps_min, args.eps_max)
    if rng.random() < args.aniso_prob:
        ratio = rng.uniform(1.0, args.aniso_max)          # elongation factor
        if rng.random() < 0.5:
            eps_x, eps_y = eps * ratio, eps
        else:
            eps_x, eps_y = eps, eps * ratio
    else:
        eps_x = eps_y = eps
    # log-uniform time -> spread of coarsening length scales
    t_end = float(np.exp(rng.uniform(np.log(args.t_min), np.log(args.t_max))))
    return dict(c0=float(c0), eps_x=float(eps_x), eps_y=float(eps_y),
                t_end=t_end)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    start = time.time()
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--nb-grid-pts", type=int, nargs=2, default=[64, 64])
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000,
                   help="RNG seed (use one disjoint from your training seeds)")
    p.add_argument("--out-dir", default="cahn_hilliard_test")
    # CH parameters
    p.add_argument("--c0-min", type=float, default=-0.5,
                   help="min mean composition (vf ~ (1+c0)/2 -> 0.25)")
    p.add_argument("--c0-max", type=float, default=0.5,
                   help="max mean composition (vf ~ 0.75)")
    p.add_argument("--eps-min", type=float, default=0.8,
                   help="min interface width [pixels]")
    p.add_argument("--eps-max", type=float, default=1.6,
                   help="max interface width [pixels]")
    p.add_argument("--aniso-prob", type=float, default=0.5,
                   help="fraction of samples with anisotropic gradient energy")
    p.add_argument("--aniso-max", type=float, default=3.0,
                   help="max eps_x/eps_y ratio for anisotropic samples")
    p.add_argument("--t-min", type=float, default=50.0)
    p.add_argument("--t-max", type=float, default=800.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--min-vf", type=float, default=0.05,
                   help="reject samples whose thresholded vf is below this")
    p.add_argument("--max-vf", type=float, default=0.95)
    # homogenization -- keep identical to the training data generation!
    p.add_argument("--kappa-solid", type=float, default=1.0)
    p.add_argument("--penalty", type=float, default=2.0)
    p.add_argument("--void-ratio", type=float, default=1e-8,
                   help="must match what batch_parallelization passed to "
                        "simulate_conduction.py")
    p.add_argument("--element", choices=["p1", "q1"], default="p1")
    p.add_argument("--cg-tol", type=float, default=1e-8)
    p.add_argument("--cg-maxiter", type=int, default=5000)
    p.add_argument("--no-homogenize", action="store_true",
                   help="only generate structures (no muGrid needed); "
                        "writes .npy instead of .nc")
    p.add_argument("--preview", action="store_true",
                   help="save a PNG mosaic of the first 16 structures")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    nb = tuple(args.nb_grid_pts)
    batch_dir = os.path.join(
        args.out_dir, f"res_{nb[0]}", f"batch_00_seed{args.seed}")
    os.makedirs(batch_dir, exist_ok=True)

    homog = None
    if not args.no_homogenize:
        import muGrid
        from muTopOpt import HomogenizationConductivity, SimpConductivity
        comm = muGrid.Communicator()       # serial: CH test set is small
        material = SimpConductivity(args.kappa_solid, args.penalty,
                                    args.void_ratio)
        homog = HomogenizationConductivity(
            nb, material, comm=comm, element=args.element,
            preconditioner="green-jacobi",
            cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
            dtype=np.float64)

    metadata = []
    previews = []
    run_id = 0
    attempts = 0
    while run_id < args.num_samples:
        attempts += 1
        if attempts > 20 * args.num_samples:
            print("too many rejected samples -- check c0/vf ranges")
            break
        t0 = time.time()
        prm = sample_params(rng, args)
        c = cahn_hilliard(nb, prm["c0"], prm["eps_x"], prm["eps_y"],
                          prm["t_end"], args.dt, rng)
        rho = (c >= 0.0).astype(np.float64)
        vf = float(rho.mean())
        if not (args.min_vf <= vf <= args.max_vf):
            continue

        if len(previews) < 16:
            previews.append(rho)

        name = f"result2D_{run_id:03d}"
        entry = dict(run_id=run_id, volume_fraction=vf, source="cahn_hilliard",
                     **prm)

        if homog is None:
            np.save(os.path.join(batch_dir, name + ".npy"), rho)
            entry["output_file"] = name + ".npy"
        else:
            homog.set_density(rho)
            kappa = np.asarray(homog.homogenized_tangent()).reshape(2, 2)
            kflat = [float(x) for x in kappa.reshape(-1)]

            out = os.path.join(batch_dir, name + ".nc")
            field = homog.scalar_field("density")
            field.p[...] = homog.to_device(rho)
            fio = muGrid.FileIONetCDF(
                out, muGrid.FileIONetCDF.OpenMode.Overwrite, comm)
            fio.register_field_collection(homog.fc, field_names=["density"])
            fio.write_global_attribute(
                "domain_lengths", [float(x) for x in homog.domain_lengths])
            fio.write_global_attribute("nb_grid_pts", [int(n) for n in nb])
            fio.write_global_attribute("command_line", shlex.join(sys.argv))
            fio.write_global_attribute("source", "cahn_hilliard")
            fio.write_global_attribute(
                "ch_params", [prm["c0"], prm["eps_x"], prm["eps_y"],
                              prm["t_end"]])
            fio.write_global_attribute("converged", [1])
            # rho is already binary, so grey == thresholded here
            fio.write_global_attribute("kappa_eff", kflat)
            fio.write_global_attribute("homog_data", kflat)
            fio.write_global_attribute("kappa_thresh", kflat)
            fio.append_frame().write(["density"])
            fio.close()

            entry.update(output_file=name + ".nc", converged=1,
                         kappa_xx=kflat[0], kappa_xy=kflat[1],
                         kappa_yy=kflat[3],
                         # loader compatibility: no target exists for CH,
                         # so "target" = achieved
                         target_kappa_xx=kflat[0], target_kappa_yy=kflat[3])

        entry["runtime"] = time.time() - t0
        metadata.append(entry)
        k_str = (f"kxx={entry['kappa_xx']:.3f} kyy={entry['kappa_yy']:.3f} "
                 f"kxy={entry['kappa_xy']:+.3f}") if homog else ""
        print(f"[{run_id:03d}] vf={vf:.3f} c0={prm['c0']:+.2f} "
              f"eps=({prm['eps_x']:.2f},{prm['eps_y']:.2f}) "
              f"t={prm['t_end']:.0f} {k_str}")
        run_id += 1

    with open(os.path.join(batch_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if args.preview and previews:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = int(np.ceil(np.sqrt(len(previews))))
        fig, axs = plt.subplots(k, k, figsize=(2 * k, 2 * k))
        for ax in np.ravel(axs):
            ax.axis("off")
        for i, (ax, r) in enumerate(zip(np.ravel(axs), previews)):
            ax.imshow(r.T, origin="lower", cmap="gray_r", vmin=0, vmax=1)
            m = metadata[i]
            ttl = f"vf={m['volume_fraction']:.2f}"
            if "kappa_xx" in m:
                ttl += f"\nkxx={m['kappa_xx']:.2f} kyy={m['kappa_yy']:.2f}"
            ax.set_title(ttl, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(batch_dir, "preview.png"), dpi=120)

    print(f"wrote {len(metadata)} samples to {batch_dir} "
          f"in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
