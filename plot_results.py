import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('build/benchmark_results_hd.csv')

# Create figure with two subplots
plt.figure(figsize=(12, 5))

# Plot speedup
plt.subplot(1, 2, 1)
plt.plot(df['k'], df['speedup'], 'b-o')
plt.title('GPU Speedup (Full HD Image)')
plt.xlabel('K (number of clusters)')
plt.ylabel('Average Speedup (times faster)')
plt.grid(True)

# Plot PSNR
plt.subplot(1, 2, 2)
plt.plot(df['k'], df['psnr'], 'r-o')
plt.title('PSNR (Full HD Image)')
plt.xlabel('K (number of clusters)')
plt.ylabel('Average PSNR (dB)')
plt.grid(True)

plt.tight_layout()
plt.savefig('kmeans_analysis.png')