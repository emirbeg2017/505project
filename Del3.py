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

"""
Question 3:
3.	Is there an association between gender and race (Black vs. Asian) in the population of Queens?
    a.	H0: Sex and race (Black/Asian) are independent. The proportion of males and females is the same across the racial groups.
    b.	H1: Sex and race (Black/Asian) are dependent. The proportion of males and females differs across the racial groups.

"""

race_sex_counts = df[df['ARREST_BORO'] == 'Q'].groupby(['PERP_RACE', 'PERP_SEX']).size().unstack()
print('\n\n\nQuestion 3:')
black_asian_df = race_sex_counts.loc[['BLACK', 'ASIAN / PACIFIC ISLANDER']]
male_counts = black_asian_df['M']
female_counts = black_asian_df['F']


male_arrests = male_counts.to_dict()
female_arrests = female_counts.to_dict()


# Create the contingency table
contingency_table = np.array([
    [male_arrests['BLACK'], male_arrests['ASIAN / PACIFIC ISLANDER']],  # Male (row 1)
    [female_arrests['BLACK'], female_arrests['ASIAN / PACIFIC ISLANDER']]  # Female (row 2)
])

# Perform Chi-Square Test for Independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Print results
print(expected)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")


# Interpretation
alpha = 0.05  # Significance level
if p < alpha:
    print("P is less than alpha so:")
    print("Reject the null hypothesis (H0): Sex and race (Black/Asian) are dependent. The proportion of males and females differs across the racial groups.")
else:
    print("P is greater than alpha so:")
    print("Fail to reject the null hypothesis (H0): Sex and race (Black/Asian) are independent. The proportion of males and females is the same across the racial groups.")
   
# Create Race-Gender Bar Plot
# Extract expected values for each race and gender
expected_male_black = expected[0][0]
expected_male_asian = expected[0][1]
expected_female_black = expected[1][0]
expected_female_asian = expected[1][1]

# Extract observed values for male and female arrests
races = list(male_arrests.keys())
male_values = list(male_arrests.values())
female_values = list(female_arrests.values())
expected_male_values = [expected_male_black, expected_male_asian]
expected_female_values = [expected_female_black, expected_female_asian]

# Set bar width and positions
width = 0.35  # Width of bars
x = np.arange(len(races))  # x-axis positions for the races

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the observed values (actual arrests) on the left side
ax.bar(x - width/2, female_values, width, label='Observed Female Arrests', color='orange')
ax.bar(x - width/2, male_values, width, bottom=female_values, label='Observed Male Arrests', color='blue')

# Plot the expected values (expected arrests) on the right side
ax.bar(x + width/2, expected_female_values, width, label='Expected Female Arrests', color='lightcoral', alpha=0.7)
ax.bar(x + width/2, expected_male_values, width, bottom=expected_female_values, label='Expected Male Arrests', color='lightgreen', alpha=0.7)

# Labels and title
ax.set_xlabel("Race")
ax.set_ylabel("Number of Arrests")
ax.set_title("Male vs. Female Arrests by Race (Observed and Expected) in Queens for Black and Asian/Pacific Islander")
ax.set_xticks(x)
ax.set_xticklabels(races)
ax.legend()

# Display values on top of bars
for i, race in enumerate(races):
    ax.text(i - width/2, female_values[i] / 2, f"{int(female_values[i])}", ha='center', color='black', fontsize=10)
    ax.text(i - width/2, female_values[i] + male_values[i] / 2, f"{int(male_values[i])}", ha='center', color='black', fontsize=10)
    ax.text(i + width/2, expected_female_values[i] / 2, f"{int(expected_female_values[i])}", ha='center', color='black', fontsize=10)
    ax.text(i + width/2, expected_female_values[i] + expected_male_values[i] / 2, f"{int(expected_male_values[i])}", ha='center', color='black', fontsize=10)

plt.tight_layout()  # To prevent overlapping of labels
plt.show()
