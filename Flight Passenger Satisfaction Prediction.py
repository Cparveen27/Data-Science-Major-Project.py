
# Step-by-step analysis and ML pipeline for airline satisfaction dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load the cleaned dataset
#df = pd.read_csv("df = pd.read_csv("C:\\Users\\cprve\\Downloads\\cleaned_test.csv")")
df = pd.read_csv("C:/Users/cprve/Downloads/cleaned_test.csv")


# 2. Data Cleaning / Preprocessing
# ---------------------------------------------------

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize categorical text columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip().str.lower()

# Ensure correct data types
df['satisfaction'] = df['satisfaction'].astype('category')
df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].astype(int)

# 3. Split into numerical and categorical features
# ---------------------------------------------------
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('id')  # Remove ID if present
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features.remove('satisfaction')  # Exclude target variable

# 4. EDA
# ---------------------------------------------------
# 4.1 Countplot and Piecharts for categorical features
for col in categorical_features:
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x=col)
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=45)
    plt.show()

    df[col].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6), title=f'Pie Chart of {col}')
    plt.ylabel('')
    plt.show()

# 4.2 Barplots of categorical vs target
for col in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col, hue='satisfaction')
    plt.title(f'{col} vs Satisfaction')
    plt.xticks(rotation=45)
    plt.show()

# 4.3 KDE plots for numerical features
for col in numerical_features[:4]:
    plt.figure(figsize=(8, 4))
    df[col].plot(kind='hist', bins=30, density=True, alpha=0.6)
    df[col].plot(kind='kde')
    plt.title(f'Distribution and KDE of {col}')
    plt.show()

# 4.4 Barplots of numerical vs target (mean)
for col in numerical_features[:4]:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x='satisfaction', y=col, estimator=np.mean)
    plt.title(f'{col} vs Satisfaction (Mean)')
    plt.show()

# 5. Heatmap for correlations
plt.figure(figsize=(15, 10))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# 6. Outlier Handling (optional - using IQR method on selected columns)
for col in ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 7. Separate features and target
X = df.drop(columns=['satisfaction', 'id'])
y = df['satisfaction']

# 8. Encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# 9. Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 10. Apply ML Algorithms and Evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))

# 11. Select the best model (based on F1-score or overall performance)

# 12. Predict on test data (if available, apply the same preprocessing steps)
# Example:
# test_df = pd.read_csv("your_test_file.csv")
# test_encoded = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)
# predictions = best_model.predict(test_encoded)
