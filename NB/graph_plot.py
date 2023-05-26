import pandas as pd
import matplotlib.pyplot as plt

# Read the file line by line
with open('NB/parametersData/NB_grid_results_recall.txt', 'r') as f:
    lines = f.readlines()

# Process the lines
data = []
for line in lines:
    line = line.strip()
    parts = line.split(', ')
    alpha = float(parts[0].split(': ')[1])
    mean_test_score = float(parts[1].split(': ')[1])
    std_test_score = float(parts[2].split(': ')[1])
    data.append([alpha, mean_test_score, std_test_score])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Alpha', 'Mean Test Score', 'Std Test Score'])

# Set Alpha as the index (optional)
df.set_index('Alpha', inplace=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Mean Test Score'], label='Recall')
plt.xlabel('Alpha')
plt.ylabel('Recall')
plt.title('NB tinklelio paieškos rezultatai')
plt.legend()
plt.grid()

# Save the figure
plt.savefig('NB/parametersData/Graph.png')

plt.show()
