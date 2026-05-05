import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("D:\\ps\\HouseaPricePrediction\\house_prices_practice.csv")

# Handle missing values (IMPORTANT)
df = df.dropna()

# Select features and target
X = df[['LotArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt',
        'TotalBsmtSF', 'GarageCars', 'GrLivArea', 'OverallQual']]
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Linear Regression --------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Linear Regression:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# -------- Random Forest --------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest:")
print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))

# -------- Save BEST model --------
if r2_score(y_test, y_pred_rf) > r2_score(y_test, y_pred):
    pickle.dump(rf, open('model.pkl', 'wb'))
    print("\nRandom Forest model saved")
else:
    pickle.dump(model, open('model.pkl', 'wb'))
    print("\nLinear Regression model saved")