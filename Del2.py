#Imports
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Load CSV file into a DataFrame
df = pd.read_csv("NYPD_Arrest_Data__Year_to_Date.csv")


# Arrest by Borough  bar graph
filtered_df = df['ARREST_BORO'].value_counts()

labels_int = filtered_df.index.tolist()
labels = list(map(str,labels_int))
values = filtered_df.values

# print(filtered_df)
#palette
colors = sns.color_palette("Set2", len(labels))
plt.figure(figsize=(17, 6))
plt.bar(labels,values,color = colors)

plt.title('Barplot of arrest Boroughs')
plt.xlabel('Boroughs (K = Brooklyn, M = Manhattan, B = Bronx, Q = Queens, S = Staten Island)')
plt.ylabel('Number of Arrests')

plt.show()

# Federal Holidays
# Holiday dates and names
f_hols = pd.to_datetime(['1/1/2024', '1/15/2024', '2/19/2024', '5/27/2024',
                         '6/19/2024', '7/4/2024', '9/2/2024'])
holiday_names = ['New Year', 'Martin Luther King Jr. Day', 'Presidents’ Day', 'Memorial Day', 
                 'Juneteenth', 'Independence Day', 'Labor Day']

# Map holiday dates to names in a dictionary
holiday_dict = dict(zip(f_hols, holiday_names))

#date_time format
df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'])

# Count occurrences of each arrest date
filtered_df = df['ARREST_DATE'].value_counts().sort_index()

#Some Stats
print('Max:', filtered_df.max(), filtered_df.idxmax())
print('Min:', filtered_df.min(), filtered_df.idxmin())
print('Mode:', filtered_df.mode(), filtered_df[filtered_df.isin(filtered_df.mode())])
print('Average: ', filtered_df.mean())
print('Standard_deviation: ', filtered_df.std())
print('Q1: ', filtered_df.quantile(0.25))
print('Q2 (median): ', filtered_df.quantile(0.50))
print('Q3: ', filtered_df.quantile(0.75))


# Compute Q1, Q3, and IQR
Q1 = filtered_df.quantile(0.25)
Q3 = filtered_df.quantile(0.75)
IQR = Q3 - Q1

#lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = filtered_df[(filtered_df < lower_bound) | (filtered_df > upper_bound)]

print(upper_bound, lower_bound)

labels = filtered_df.index
values = filtered_df.values
#Make plot larger in size
plt.figure(figsize=(25, 6))

# Plot the arrests over time
plt.plot(labels, values, marker='x', linestyle='-', color='b', label="Arrests")

# Add vertical dashed lines for holidays
for date in f_hols:
    plt.axvline(pd.Timestamp(date), color='red', linestyle='dashed', linewidth=1.5, 
                label='Holiday' if date == f_hols[0] else "")



plt.xticks(f_hols, holiday_names, rotation=90)

plt.xlabel('Arrest Date')
plt.ylabel('Number of Arrests')
plt.title('Line Plot of Arrests Over Time')

plt.legend()
plt.show()


#Boxplot for arrest counts by month
#Map holiday dates to names
holiday_dict = dict(zip(f_hols, holiday_names))


df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'])

#Date formatting getting month and date in seperate columns
#.dt.month month seperated
df['Month'] = df['ARREST_DATE'].dt.month

#Get all the dates consolidated with counts
#.groupby consolidates the days, and .transform('count') will count how many were there AND associates it with the date
df['ARREST_COUNT'] = df.groupby('ARREST_DATE')['ARREST_DATE'].transform('count')

plt.figure(figsize=(12, 6))

custom_palette = sns.color_palette("Set3", n_colors=9)

sns.boxplot(x='Month', y='ARREST_COUNT', data=df, palette=custom_palette, hue = 'Month', legend = False)

plt.xticks(ticks=range(9), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])

plt.xlabel('Month')
plt.ylabel('Number of Arrests')
plt.title('Boxplot of Arrests by Month')
plt.show()


#Pie Chart
# Get the race counts for queens
race_counts = df[df['ARREST_BORO'] == 'Q']['PERP_RACE'].value_counts()

print('Max:', race_counts.max(), race_counts.idxmax())
print('Min:', race_counts.min(), race_counts.idxmin())
print('Average: ', race_counts.mean())
print('Standard_deviation: ', race_counts.std())
print('Q1: ', race_counts.quantile(0.25))
print('Q2 (median): ', race_counts.quantile(0.50))
print('Q3: ', race_counts.quantile(0.75))

Q1 = race_counts.quantile(0.25)
Q3 = race_counts.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = race_counts[(race_counts < lower_bound) | (race_counts > upper_bound)]

print(upper_bound, lower_bound)

labels_int = race_counts.index.tolist()
labels = list(map(str, labels_int))
sizes = race_counts.values / sum(race_counts.values) 

# Define colors
colors = ['red', 'green', 'peru', 'hotpink', 'aliceblue', 'steelblue', 'yellow']


explode = [0.2 if count < race_counts.max() * 0.2 else 0 for count in race_counts]

plt.figure(figsize=(6,6))
wedges, _ = plt.pie(sizes,
                    labels=[''] * len(race_counts), 
                    colors=colors,
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'},
                    explode=explode)


legend_labels = [f"{race}: {perc:.1f}%" for race, perc in zip(race_counts.index, sizes * 100)]

# Add a legend
plt.legend(wedges, legend_labels, title="Perp Race", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)


plt.title("Proportion of Arrests in Queens by Perp Race")
plt.show()

race_sex_counts = df[df['ARREST_BORO'] == 'Q'].groupby(['PERP_RACE', 'PERP_SEX']).size().unstack()

race_sex_counts.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.xlabel('Race')
plt.ylabel('Count')
plt.title('Arrests by Race and Sex in Queens')
plt.legend(title='Sex')
plt.xticks(rotation=90)
plt.show()

print(race_sex_counts)