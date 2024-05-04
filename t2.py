import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
# Generating synthetic data
np.random.seed(0)
n_samples = 1000

# Features: income, schools, hospitals, crime rates
income = np.random.normal(50000, 10000, n_samples)
schools = np.random.normal(8, 2, n_samples)
hospitals = np.random.normal(5, 1, n_samples)
crime_rates = np.random.normal(3, 1, n_samples)

# Target: home prices
home_prices = 100000 + 5000*income + 2000*schools + 5000*hospitals - 10000*crime_rates + np.random.normal(0, 10000, n_samples)

# Creating DataFrame
data = pd.DataFrame({'Income': income, 'Schools': schools, 'Hospitals': hospitals, 'CrimeRates': crime_rates, 'HomePrices': home_prices})

# Splitting data into train and test sets
X = data[['Income', 'Schools', 'Hospitals', 'CrimeRates']]
y = data['HomePrices']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
