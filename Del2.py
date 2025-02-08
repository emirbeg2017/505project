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
#palette
colors = sns.color_palette("Set2", len(labels))

plt.bar(labels,values,color = colors)

plt.title('Barplot of arrest Boroughs')
plt.xlabel('Boroughs (K = Brooklyn, M = Manhattan, B = Bronx, Q = Queens, S = Staten Island)')
plt.ylabel('Number of Arrests')

plt.show()

# Federal Holidays
# side by side boxplots histogram data on holidays and nonholidays
f_hols = ['1/1/2024', '1/15/2024', '2/19/2024', '5/27/2024','6/19/2024', '7/4/2024', '9/2/2024']

f_hols = pd.to_datetime(['1/1/2024', '1/15/2024', '2/19/2024', '5/27/2024',
                         '6/19/2024', '7/4/2024', '9/2/2024'])

dates_df = pd.to_datetime(df['ARREST_DATE'])

plt.figure(figsize=(12, 6))


plt.hist(dates_df, bins = 36, edgecolor='black', alpha=0.7)

for date in f_hols:
    plt.axvline(date, color='red', linestyle='dashed', linewidth=1.5, label='Holiday' if date == f_hols[0] else "")

plt.xlabel('Arrest Date')
plt.ylabel('Number of Arrests')
plt.title('Histogram of Arrests Over Time')
plt.xticks(rotation=45)  # Rotate by 45 degrees

plt.show

# Get the race counts for 'Q' (Queens)
race_counts = df[df['ARREST_BORO'] == 'Q']['PERP_RACE'].value_counts()

# Prepare labels and sizes
labels_int = race_counts.index.tolist()
labels = list(map(str, labels_int))
sizes = race_counts.values / sum(race_counts.values)  # Normalize to get proportions

# Define colors
colors = ['red', 'green', 'peru', 'hotpink', 'aliceblue', 'steelblue', 'yellow']

# Create the explode list
explode = [0.2 if count < race_counts.max() * 0.2 else 0 for count in race_counts]

# Create the pie chart without percentages in the pie
plt.figure(figsize=(6,6))
wedges, _ = plt.pie(sizes,
                    labels=[''] * len(race_counts), 
                    colors=colors,
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'},
                    explode=explode)

# Manually calculate percentages and create legend labels
legend_labels = [f"{race}: {perc:.1f}%" for race, perc in zip(race_counts.index, sizes * 100)]

# Add a legend BELOW the pie chart with percentages
plt.legend(wedges, legend_labels, title="Perp Race", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

# Add a title
plt.title("Proportion of Arrests in Queens by Perp Race")

# Show the plot
plt.show()