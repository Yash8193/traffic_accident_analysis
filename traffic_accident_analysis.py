

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Collection – Simulate Dataset
np.random.seed(0)
n = 100
data = {
    'Accident_Severity': np.random.choice([1, 2, 3], size=n),
    'Day_of_Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=n),
    'Light_Conditions': np.random.choice(['Daylight', 'Darkness - lights lit', 'Darkness - lights unlit'], size=n),
    'Weather_Conditions': np.random.choice(['Fine', 'Rain', 'Snow', 'Fog', 'Wind'], size=n),
    'Road_Surface_Conditions': np.random.choice(['Dry', 'Wet', 'Snow', 'Ice'], size=n),
    'Number_of_Vehicles': np.random.randint(1, 6, size=n),
    'Number_of_Casualties': np.random.randint(0, 5, size=n)
}
df = pd.DataFrame(data)

# 2. Insert Missing Values
for col in ['Weather_Conditions', 'Light_Conditions']:
    df.loc[df.sample(frac=0.1).index, col] = np.nan

# 3. Handle Missing Values (fixed without inplace)
df['Weather_Conditions'] = df['Weather_Conditions'].fillna(df['Weather_Conditions'].mode()[0])
df['Light_Conditions'] = df['Light_Conditions'].fillna(df['Light_Conditions'].mode()[0])

# 4. Feature Engineering
severity_map = {1: 'High', 2: 'Medium', 3: 'Low'}
df['Severity_Level'] = df['Accident_Severity'].map(severity_map)

# 5. Handle Outliers in Vehicles
Q1 = df['Number_of_Vehicles'].quantile(0.25)
Q3 = df['Number_of_Vehicles'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Number_of_Vehicles'] >= Q1 - 1.5 * IQR) & (df['Number_of_Vehicles'] <= Q3 + 1.5 * IQR)]

# 6. Summary Statistics
print("Summary:\n", df.describe(include='all'))

# 7. Visualizations

# Accident Severity Distribution
sns.countplot(x='Severity_Level', data=df)
plt.title("Accident Severity Distribution")
plt.savefig("severity_plot.png")
plt.clf()

# Accidents by Day and Severity
sns.countplot(x='Day_of_Week', hue='Severity_Level', data=df)
plt.xticks(rotation=45)
plt.title("Accidents by Day and Severity")
plt.tight_layout()
plt.savefig("daywise_plot.png")
plt.clf()
