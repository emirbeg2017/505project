#Imports
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Load CSV file into a DataFrame
df = pd.read_csv("NYPD_Arrest_Data__Year_to_Date.csv")

"""
First question:
1.	Are arrests in Bronx and Brooklyn proportional to their populations?
    a.	H0: Arrests in the Bronx and Brooklyn are proportional to their populations
    b.	H1: Arrests in the Bronx and Brooklyn are not proportional to their populations
"""
# Arrest by Borough  bar graph
filtered_df = df['ARREST_BORO'].value_counts()

labels_int = filtered_df.index.tolist()
labels = list(map(str,labels_int))
values = filtered_df.values

# 2679620 according to google is the population of Brooklyn
# 1419250 according to google is the population of Bronx
arrests = {'K': filtered_df['K'], 'B': filtered_df['B']}
population = {'K': 2679620, 'B': 1419250}

# Total values
total_population = sum(population.values())
total_arrests = sum(arrests.values())

# Compute expected arrests based on population proportion
expected_arrests = {boro: (pop / total_population) * total_arrests for boro, pop in population.items()}

# Convert to numpy arrays for Chi-Square test
observed = np.array(list(arrests.values()))
expected = np.array(list(expected_arrests.values()))

# Run Chi-Square Goodness of Fit Test
chi2, p = stats.chisquare(observed, expected)

# Print results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")

# Convert to lists for plotting
boroughs = list(arrests.keys())
observed_values = list(arrests.values())
expected_values = list(expected_arrests.values())

# Set up bar width and positions
x = np.arange(len(boroughs))  # X-axis positions
width = 0.4  # Width of bars

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, observed_values, width, label='Observed Arrests', color='blue')
bars2 = ax.bar(x + width/2, expected_values, width, label='Expected Arrests', color='orange')

# Labels and title
ax.set_xlabel("Borough")
ax.set_ylabel("Number of Arrests")
ax.set_title("Observed vs. Expected Arrests in Bronx & Brooklyn")
ax.set_xticks(x)
ax.set_xticklabels(boroughs)
ax.legend()

# Display values on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)

plt.show()


# Interpretation
alpha = 0.05  # Significance level
if p < alpha:
    print("Reject the null hypothesis: Arrests are NOT proportional to population.")
else:
    print("Fail to reject the null hypothesis: Arrests MAY BE proportional to population.")
