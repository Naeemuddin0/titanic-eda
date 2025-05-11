
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (ensure 'train.csv' is in the working directory)
try:
    df = pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please ensure the file is in the working directory.")
    raise

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nData Info:\n")
df.info()
print("\nSummary Statistics:\n", df.describe())

# Check missing values before
print("\nMissing Values Before:\n", df.isnull().sum())

# Handle missing values
df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)
df.drop('Cabin', axis=1, inplace=True, errors='ignore')

# Check missing values after
print("\nMissing Values After:\n", df.isnull().sum())

# Explore categorical variables
plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.savefig('survival_count.png')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
plt.title('Survival Rate by Class and Sex')
plt.savefig('survival_by_class_sex.png')
plt.show()

# Explore numerical variables
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival')
plt.savefig('fare_by_survival.png')
plt.show()

# Find relationships
plt.figure(figsize=(12, 8))
numerical_df = df.select_dtypes(include=['number'])
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Summary of insights
print("\nKey Insights:")
print("- First-class passengers had a higher survival rate.")
print("- Females were more likely to survive than males.")
print("- Younger passengers and those paying higher fares had better survival chances.")