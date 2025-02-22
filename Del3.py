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

print(arrests)

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

# Interpretation
alpha = 0.05  # Significance level
if p < alpha:
    print("Reject the null hypothesis: Arrests are NOT proportional to population.")
else:
    print("Fail to reject the null hypothesis: Arrests MAY BE proportional to population.")
    
plt.show()





"""
Question 2:
2.  Are there more arrests on Federal holidays?
    a.	H0: There is no difference in the number of arrests on federal holidays and non-federal holidays
    b.	H1: There is a difference in the number of arrests on federal holidays and non-federal holidays
    
"""

f_hols = pd.to_datetime(['1/1/2024', '1/15/2024', '2/19/2024', '5/27/2024',
                         '6/19/2024', '7/4/2024', '9/2/2024'])
holiday_names = ['New Year', 'Martin Luther King Jr. Day', 'Presidents’ Day', 'Memorial Day', 
                 'Juneteenth', 'Independence Day', 'Labor Day']

# Convert ARREST_DATE to datetime format
df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'])

# Count occurrences of each arrest date
filtered_df = df['ARREST_DATE'].value_counts().sort_index().reset_index()
filtered_df.columns = ['ARREST_DATE', 'COUNT']

# Add a column to indicate whether the date is a holiday
filtered_df['IS_HOLIDAY'] = filtered_df['ARREST_DATE'].isin(f_hols).map({True: 'Yes', False: 'No'})


holiday_arrests = filtered_df[filtered_df['IS_HOLIDAY'] == 'Yes'][['ARREST_DATE', 'COUNT']].values.tolist()
non_holiday_arrests = filtered_df[filtered_df['IS_HOLIDAY'] == 'No'][['ARREST_DATE', 'COUNT']].values.tolist()

# Extract only the counts
holiday_counts = [count for _, count in holiday_arrests]
non_holiday_counts = [count for _, count in non_holiday_arrests]

print('\n')
print('\n')
print('Question 2:')

var_holiday = np.var(holiday_counts, ddof=1)  # Sample variance
var_non_holiday = np.var(non_holiday_counts, ddof=1)

print(f"Variance of holiday arrests: {var_holiday}")
print(f"Variance of non-holiday arrests: {var_non_holiday}")

t_stat, p = stats.ttest_ind(holiday_counts, non_holiday_counts, equal_var=False)

# Q-Q plot for holiday arrests
plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1) 
stats.probplot(holiday_counts, dist="norm", plot=plt)
plt.title("Q-Q Plot for Federal Holiday Arrests")


plt.subplot(1, 2, 2) 
stats.probplot(non_holiday_counts, dist="norm", plot=plt)
plt.title("Q-Q Plot for Non-Holiday Arrests")

plt.tight_layout()
plt.show()




print(f"T-Statistic: {t_stat}")
print(f"P-Value: {p}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: The number of arrests is significantly different on Federal holidays.")
else:
    print("Fail to reject the null hypothesis: No significant difference in the number of arrests on holidays vs. non-holidays.")
