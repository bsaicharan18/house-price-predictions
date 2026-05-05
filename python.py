import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  

# Load dataset
df = pd.read_csv("D:\\ps\\HouseaPricePrediction\\house_prices_practice.csv")

# Basic info
print(df.head())
print(df.columns)

# Handle missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

# ------------------ Visualization ------------------

features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt',
            'TotalBsmtSF', 'GarageCars', 'GrLivArea', 'OverallQual']

# Plot each feature vs SalePrice
for feature in features:
    sns.scatterplot(x=feature, y='SalePrice', data=df)
    plt.title(f"{feature} vs SalePrice")
    plt.show()

# Histogram
sns.histplot(df['SalePrice'], kde=True)
plt.title("SalePrice Distribution")
plt.show()

# Correlation Heatmap
corr = df[features + ['SalePrice']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ------------------ MODEL BUILDING ------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ALL FEATURES USED HERE 💥
X = df[features]
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Linear Regression --------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Linear Regression Results:")
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))

# -------- Random Forest --------
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Results:")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))