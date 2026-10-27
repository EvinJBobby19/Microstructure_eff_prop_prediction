
import glob
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Change this path to your dataset location
DATA_GLOB = "//wsl.localhost/Ubuntu/home/evin/muTopOpt/test_dataset_5000/**/result2D_*.nc"

kxx, kxy, kyy = [], [], []
n_failed = 0

files = sorted(glob.glob(DATA_GLOB, recursive=True))
print(f"Found {len(files)} files")

for file in files:
    try:
        with Dataset(file, "r") as ds:
            if "kappa_thresh" not in ds.ncattrs():
                n_failed += 1
                continue

            # Optional: use the same convergence filter as your CNN
            if "converged" in ds.ncattrs():
                if not bool(np.asarray(ds.getncattr("converged")).ravel()[0]):
                    continue

            K = np.asarray(ds.getncattr("kappa_thresh")).reshape(2, 2)

            kxx.append(K[0, 0])
            kxy.append((K[0, 1] + K[1, 0]) / 2)
            kyy.append(K[1, 1])

    except Exception as e:
        n_failed += 1
        print(f"Failed: {file}: {e}")

kxx = np.array(kxx)
kxy = np.array(kxy)
kyy = np.array(kyy)

if len(kxx) == 0:
    raise RuntimeError("No conductivity values loaded. Check DATA_GLOB.")

print(f"\nLoaded {len(kxx)} samples")
print(f"Unreadable/missing attribute: {n_failed}")

def report(name, values):
    print(f"\n{name}")
    print(f"Min: {values.min():.6g}")
    print(f"Max: {values.max():.6g}")
    print(f"Mean: {values.mean():.6g}")
    print(f"Median: {np.median(values):.6g}")

    for threshold in [1e-6, 0.001, 0.01, 0.05, 0.1]:
        count = np.sum(values < threshold)
        percentage = 100 * count / len(values)
        print(f"  Values < {threshold:g}: {count} ({percentage:.2f}%)")

report("kxx", kxx)
report("kyy", kyy)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for ax, values, name in zip(
    axes, [kxx, kxy, kyy], ["kxx", "kxy", "kyy"]
):
    ax.hist(values, bins=50, color="steelblue", edgecolor="black")
    ax.set_title(name)
    ax.set_xlabel("Thresholded effective conductivity")
    ax.set_ylabel("Number of samples")
    ax.grid(alpha=0.2)

fig.suptitle(f"Distribution of kappa_thresh ({len(kxx)} samples)")
plt.tight_layout()
plt.savefig("kappa_distribution.png", dpi=200)
plt.show()
