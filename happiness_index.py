# -*- coding: utf-8 -*-
"""happiness_index.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QLhdsWHaMTwjkugVnsQc1d4WwMkJFnUy
"""

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.metrics import mean_absolute_error
# df = pd.read_csv("/content/drive/MyDrive/Happiness Index/happiness_index_report.csv")
df = pd.read_csv('C:\\Users\\Chaytu\\OneDrive\\Desktop\\DAV\\happiness_index_report.csv')
df1 = pd.read_csv('C:\\Users\\Chaytu\\OneDrive\\Desktop\\DAV\\world_happiness_corruption.csv')
# df1 = pd.read_csv("/content/drive/MyDrive/Happiness Index/world_happiness_corruption.csv")

# Display the first few rows of the dataset
print(df.head())


# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Group by year and calculate average happiness index
avg_index_by_year = df.groupby('Year')['Index'].mean()

# Plot average happiness index over the years
plt.figure(figsize=(10, 6))
avg_index_by_year.plot(kind='line', marker='o')
plt.title('Average Happiness Index Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Happiness Index')
plt.grid(True)
plt.show()

# Top 10 happiest countries in the latest year available
latest_year = df['Year'].max()
top_10_happiest_countries = df[df['Year'] == latest_year].nlargest(10, 'Index')

# Plot top 10 happiest countries
plt.figure(figsize=(10, 6))
sns.barplot(x='Country', y='Index', data=top_10_happiest_countries)
plt.title('Top 10 Happiest Countries in {}'.format(latest_year))
plt.xlabel('Country')
plt.ylabel('Happiness Index')
plt.xticks(rotation=45, ha='right')
plt.show()

# Choose the country you want to analyze
country_name = 'India'

# Filter the dataset for the chosen country
country_df = df[df['Country'] == country_name]

# Check if data is available for all 10 years
if country_df['Year'].nunique() == 10:
    # Plot the happiness index trend for the chosen country
    plt.figure(figsize=(10, 6))
    plt.plot(country_df['Year'], country_df['Index'], marker='o')
    plt.title('Happiness Index Trend for {}'.format(country_name))
    plt.xlabel('Year')
    plt.ylabel('Happiness Index')
    plt.grid(True)
    plt.xticks(range(2013, 2024), rotation=45)
    plt.show()
else:
    print("Data is not available for all 10 years for the selected country.")

# Display the first few rows of the dataset
print(df1.head())

# Summary statistics
print(df1.describe())

# Check for missing values
print(df1.isnull().sum())

# Select only numeric columns for correlation matrix
numeric_columns = df1.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
corr_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Group by year and calculate average happiness score
avg_score_by_year = df1.groupby('Year')['happiness_score'].mean()

# Plot average happiness score over the years
plt.figure(figsize=(10, 6))
avg_score_by_year.plot(kind='line', marker='o')
plt.title('Average Happiness Score Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Happiness Score')
plt.grid(True)
plt.show()

# Plot distribution of happiness scores
plt.figure(figsize=(10, 6))
sns.histplot(df1['happiness_score'], kde=True, bins=20)
plt.title('Distribution of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot happiness score vs GDP per capita
plt.figure(figsize=(10, 6))
sns.scatterplot(x='gdp_per_capita', y='happiness_score', data=df1, color='red')
plt.title('Happiness Score vs GDP per Capita')
plt.xlabel('GDP per Capita')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.show()

#Happiness vs Freedom scatterplot

plt.figure(figsize=(10, 6))
sns.scatterplot(x='freedom', y='happiness_score', data=df1, color='green')
plt.title('Happiness Score vs freedom')
plt.xlabel('Freedom')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.show()

#Happiness vs Health scatterplot

plt.figure(figsize=(10, 6))
sns.scatterplot(x='health', y='happiness_score', data=df1, color='blue')
plt.title('Happiness Score vs health')
plt.xlabel('Health')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.show()

# Plot happiness score by continent
plt.figure(figsize=(10, 6))
sns.boxplot(x='continent', y='happiness_score', data=df1)
plt.title('Happiness Score by Continent')
plt.xlabel('Continent')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.show()


# Select features (X) and target variable (y)
X = df1[['gdp_per_capita']]  # Feature (GDP per capita)
y = df1['happiness_score']    # Target variable (Happiness score)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Linear Regression: GDP per Capita vs Happiness Score')
plt.xlabel('GDP per Capita')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.show()

# Select data for the four specific countries
countries = ['United States', 'France', 'United Kingdom', 'Russia']
df_countries = df1[df1['Country'].isin(countries)]

# Plot multi-line graph
plt.figure(figsize=(10, 6))
sns.lineplot(x='gdp_per_capita', y='happiness_score', hue='Country', data=df_countries, marker='o')
plt.title('GDP per Capita vs Happiness Score for Four Countries')
plt.xlabel('GDP per Capita')
plt.ylabel('Happiness Score')
plt.grid(True)
plt.legend(loc='best')
plt.show()

# Select data for United States and Russia
us_data = df1[df1['Country'] == 'United States']
russia_data = df1[df1['Country'] == 'Russia']

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(us_data['gdp_per_capita'], us_data['government_trust'], color='blue', label='United States')
plt.scatter(russia_data['gdp_per_capita'], russia_data['government_trust'], color='red', label='Russia')
plt.title('GDP per Capita vs Government Trust Scores')
plt.xlabel('GDP per Capita')
plt.ylabel('Government Trust Score')
plt.grid(True)
plt.legend(loc='best')
plt.show()

# List of years you want to filter
years = [2015, 2016, 2017, 2018, 2019, 2020]

# Iterate over years and save CSV files
for year in years:
    # Filter the dataframe for the current year
    year_df = df[df['Year'] == year]

    # Generate the filename
    filename = f"{year}_index.csv"

    # Save the filtered dataframe to a CSV file
    year_df.to_csv(filename, index=False)

    print(f"CSV file for {year} saved as {filename}")

# Remove the 'social_support' column from the first CSV file
df1.drop('social_support', axis=1, inplace=True)

# Sort the dataframe by 'Country' column
df1.sort_values(by='Country', inplace=True)

# Group the dataframe by year and save each group into a separate CSV file
for year in range(2015, 2021):
    year_df = df1[df1['Year'] == year]
    year_df.to_csv(f'{year}_overall.csv', index=False)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load all datasets from 2015 to 2020
dfs = []
for year in range(2015, 2021):
    df = pd.read_csv(f'{year}_overall.csv')
    dfs.append(df)

# Concatenate all datasets into one dataframe
combined_df = pd.concat(dfs, ignore_index=True)

# Features and target variable
X = combined_df[['gdp_per_capita', 'family', 'health', 'freedom', 'generosity', 'government_trust', 'dystopia_residual']]
y = combined_df['happiness_score']

# Initialize the Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y)

# Evaluate the model on the combined dataset (optional)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error on Training Data: {mse}')

# Save the trained model for future use

joblib.dump(model, 'happiness_prediction_model.pkl')

# Load the model
model = joblib.load('happiness_prediction_model.pkl')



# Assume these are the feature values for a country in 2023
input_data_2023 = [[1.48, 1.46, 0.81, 0.57, 0.32, 0.22, 2.14]]

# Make predictions
predicted_happiness_score_2023 = model.predict(input_data_2023)

print("Predicted Happiness Score for 2023:", predicted_happiness_score_2023)


# Assuming your data is stored in a pandas DataFrame called 'df'

# Descriptive statistics
descriptive_stats = df.describe()

# Print descriptive statistics
print(descriptive_stats)
# Save the descriptive statistics DataFrame to a .pkl file
descriptive_stats.to_pickle('descriptive_stats.pkl')
# Load the descriptive statistics DataFrame from the .pkl file
loaded_stats = pd.read_pickle('descriptive_stats.pkl')

# Display the loaded DataFrame
print(loaded_stats)
