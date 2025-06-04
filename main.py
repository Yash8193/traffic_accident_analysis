# -----------------------------------------------
# Traffic Accident Analysis – Complete Workflow
# Covers Review 1 & Review 2
# -----------------------------------------------

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Step 2: Load sample dataset
data = {
    "Accident_Severity": [1, 2, 3, 2, 1, 3, 1, 2],
    "Day_of_Week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"],
    "Number_of_Casualties": [2, 1, 3, 1, 4, 2, 5, 2],
    "Number_of_Vehicles": [2, 1, 3, 2, 4, 2, 5, 2],
    "Weather_Conditions": ["Rain", "Clear", "Fog", "Clear", None, "Snow", "Rain", None],
    "Light_Conditions": ["Daylight", "Darkness", "Daylight", "Darkness", "Daylight", None, "Darkness", None]
}

df = pd.DataFrame(data)

# Step 3: Data Cleaning – handle missing values
df['Weather_Conditions'] = df['Weather_Conditions'].fillna(df['Weather_Conditions'].mode()[0])
df['Light_Conditions'] = df['Light_Conditions'].fillna(df['Light_Conditions'].mode()[0])

# Step 4: Feature Engineering
df['Severity_Level'] = df['Accident_Severity'].map({1: 'High', 2: 'Medium', 3: 'Low'})

# Step 5: Summary Statistics
print("Summary Statistics:")
print(df.describe(include='all'))

# Step 6: Visualizations

# 6.1: Severity Level Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Severity_Level', hue='Severity_Level', palette='Set2', legend=False)
plt.title("Severity Level Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 6.2: Accidents by Day and Severity
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Day_of_Week', hue='Severity_Level', palette='Set3')
plt.title("Accidents by Day and Severity")
plt.xlabel("Day of Week")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6.3: Casualties by Severity – FIXED warning
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Severity_Level', y='Number_of_Casualties', hue='Severity_Level', palette='pastel', legend=False)
plt.title("Casualties by Severity")
plt.xlabel("Severity Level")
plt.ylabel("Number of Casualties")
plt.tight_layout()
plt.show()

# 6.4: Vehicles per Accident
plt.figure(figsize=(6, 4))
sns.histplot(df['Number_of_Vehicles'], bins=5, kde=True, color='coral')
plt.title("Vehicles per Accident")
plt.xlabel("Number of Vehicles")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 6.5: Heatmap – Weather vs Severity
plt.figure(figsize=(8, 5))
weather_severity = pd.crosstab(df['Weather_Conditions'], df['Severity_Level'])
sns.heatmap(weather_severity, annot=True, fmt='d', cmap='coolwarm')
plt.title("Weather vs Severity Level")
plt.xlabel("Severity Level")
plt.ylabel("Weather Condition")
plt.tight_layout()
plt.show()
