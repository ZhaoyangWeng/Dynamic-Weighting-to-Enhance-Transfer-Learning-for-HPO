import pandas as pd
import matplotlib.pyplot as plt

# Function to average DTM values for each iteration across different seeds and optimizers
def average_dtm_per_iteration_and_optimizer(data):
    return data.groupby(['iteration', 'optimizer'])['dtm'].mean().reset_index()

# Load the merged dataset file
data = pd.read_csv('cifar10_results.csv')  # Update the path to your merged dataset

# Averaging DTM values for each optimizer
average_results = average_dtm_per_iteration_and_optimizer(data)

# Plotting the line chart for ADTM values for each optimizer
plt.figure(figsize=(10, 6))

# Plot for each optimizer
for optimizer in average_results['optimizer'].unique():
    subset = average_results[average_results['optimizer'] == optimizer]
    if optimizer == 'IGCP + prior':
        plt.plot(subset['iteration'], subset['dtm'], linestyle='--', label=optimizer)
    else:
        plt.plot(subset['iteration'], subset['dtm'], label=optimizer)

plt.xlabel('Iteration')
plt.ylabel('Average DTM')
plt.title('Cifar-10')
plt.legend()
plt.grid(True)
plt.show()
