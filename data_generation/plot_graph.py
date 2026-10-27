import matplotlib.pyplot as plt

time_taken_64 = [1.82, 1.60, 1.78, 2.53]  # in minutes
time_taken_128 = [16.72, 8.43, 7.34, 8.12]  # in minutes
time_taken_256 = [11.05, 6.70, 6.50, 21.28]  # in minutes


time_taken_64_batch_2 = [1.08,1.64]
time_taken_128_batch_2 = [5.79,11.69]
time_taken_256_batch_2 = [12.51,]

# Replace these with your actual data
threads = [2,4]  # number of concurrent simulations
cores = [2,4,8,12,16,20]   
time_taken = [6.5,6.7,2.02,4.41,2.51,4.36]  # in minutes

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(cores, time_taken, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=8)

ax.set_xlabel('Number of cores', fontsize=12)
ax.set_ylabel('Time Taken for 10 samples(min)', fontsize=12)
ax.set_title('Runtime vs. Cores  (Threads:1 res:256)', fontsize=14)

ax.set_xticks(cores)  # ensures only your actual core counts show as ticks
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('256_nemo_runtime_th_1.png', dpi=300)
plt.show()