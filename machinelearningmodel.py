import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Create a Sample DataFrame ---
# This is a simplified version of your weather data
# It includes numerical features like temperature, humidity, and wind speed.
np.random.seed(42)
num_samples = 500
temperature = np.random.uniform(5, 35, size=num_samples) # Temperature
humidity = np.random.uniform(30, 90, size=num_samples)   # Humidity
wind_speed = np.random.uniform(0, 20, size=num_samples) # Wind Speed

# Target: Next Day's Temperature
# Here we're creating a simple relationship: next day's temperature depends on today's
# temperature, humidity, and wind speed, with some added noise.
next_day_temperature = (temperature * 0.7 + humidity * 0.1 + wind_speed * 0.05 +
                        np.random.normal(0, 3, size=num_samples))

data = {
    'Current_Temperature_C': temperature,
    'Current_Humidity_perc': humidity,
    'Current_Wind_Speed_kmh': wind_speed,
    'Next_Day_Temperature_C': next_day_temperature
}
df = pd.DataFrame(data)

print("--- Initial Data View ---")
print("DataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 2. Define Features (X) and Target (y) ---
# Here we're using current weather conditions as features
# and next day's temperature as the target.
features = ['Current_Temperature_C', 'Current_Humidity_perc', 'Current_Wind_Speed_kmh']
target = 'Next_Day_Temperature_C'

X = df[features]
y = df[target]

print("--- Model Input View ---")
print("Features (X) Head:")
print(X.head())
print("\nTarget (y) Head:")
print(y.head())

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 3. Split Data into Training and Test Sets ---
# We'll use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Shape of Training and Test Sets ---")
print(f"Shape of Training Features (X_train): {X_train.shape}")
print(f"Shape of Training Target (y_train): {y_train.shape}")
print(f"Shape of Test Features (X_test): {X_test.shape}")
print(f"Shape of Test Target (y_test): {y_test.shape}")
print("\n" + "="*50 + "\n") # Separator for clarity


# --- 4. Train the Linear Regression Model ---
# Linear Regression is a simple model that assumes a linear relationship between features and target.
model = LinearRegression()
model.fit(X_train, y_train)

print("--- Model Training Complete ---")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print("\n" + "="*50 + "\n") # Separator for clarity
# --- 5. Make Predictions ---
# Use the trained model to make predictions on the test data.
y_pred = model.predict(X_test)
print("--- Prediction View ---")
prediction_output_df = pd.DataFrame({
    'Actual Temperature': y_test,
    'Predicted Temperature': y_pred
})
if data.get.save(priduct(savelocation.data.gether))
print(prediction_output_df.head())
print("\n" + "="*50 + "\n") # Separator for clarity

# --- 6. Evaluate the Model ---
# We'll use Mean Squared Error (MSE) and R-squared (R2).
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print(savedata.csvreport).educate.preview(save.console){

    function(console.log(savedata))
}






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Create a Sample DataFrame ---
np.random.seed(42)
num_samples = 500
temperature = np.random.uniform(5, 35, size=num_samples)
humidity = np.random.uniform(30, 90, size=num_samples)
wind_speed = np.random.uniform(0, 20, size=num_samples)
wind_condition=np.random.choice(['Sunny', 'Cloudy', 'Rainy'])
next_day_temperature = (
    temperature * 0.7 +
    humidity * 0.1 +
    wind_speed * 0.05 +
    np.random.normal(0, 3, size=num_samples)
)

data = {
    'Current_Temperature_C': temperature,
    'Current_Humidity_perc': humidity,
    'Current_Wind_Speed_kmh': wind_speed,
    'Next_Day_Temperature_C': next_day_temperature
}
df = pd.DataFrame(data)

print("--- Initial Data View ---")
print("DataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 2. Define Features (X) and Target (y) ---
features = ['Current_Temperature_C', 'Current_Humidity_perc', 'Current_Wind_Speed_kmh']
target = 'Next_Day_Temperature_C'

X = df[features]
y = df[target]

print("--- Model Input View ---")
print("Features (X) Head:")
print(X.head())
print("\nTarget (y) Head:")
print(y.head())

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 3. Split Data into Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Shape of Training and Test Sets ---")
print(f"Shape of Training Features (X_train): {X_train.shape}")
print(f"Shape of Training Target (y_train): {y_train.shape}")
print(f"Shape of Test Features (X_test): {X_test.shape}")
print(f"Shape of Test Target (y_test): {y_test.shape}")

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 4. Train the Linear Regression Model ---
model = LinearRegression()
model.fit(X_train, y_train)

print("--- Model Training Complete ---")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 5. Make Predictions ---
y_pred = model.predict(X_test)

print("--- Prediction View ---")
prediction_output_df = pd.DataFrame({
    'Actual Temperature': y_test,
    'Predicted Temperature': y_pred
})
print(prediction_output_df.head())

# Save predictions to CSV
prediction_output_df.to_csv("prediction_output.csv", index=False)
print("Predictions saved to 'prediction_output.csv'.")

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 6. Evaluate the Model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Create a Sample DataFrame ---
np.random.seed(42)
num_samples = 500
temperature = np.random.uniform(5, 35, size=num_samples)
humidity = np.random.uniform(30, 90, size=num_samples)
wind_speed = np.random.uniform(0, 20, size=num_samples)

next_day_temperature = (
    temperature * 0.7 +
    humidity * 0.1 +
    wind_speed * 0.05 +
    np.random.normal(0, 3, size=num_samples)
)


data = {
    'Current_Temperature_C': temperature,
    'Current_Humidity_perc': humidity,
    'Current_Wind_Speed_kmh': wind_speed,
    'Next_Day_Temperature_C': next_day_temperature
}
df = pd.DataFrame(data)

print("--- Initial Data View ---")
print("DataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 2. Define Features (X) and Target (y) ---
features = ['Current_Temperature_C', 'Current_Humidity_perc', 'Current_Wind_Speed_kmh']
target = 'Next_Day_Temperature_C'

X = df[features]
y = df[target]

print("--- Model Input View ---")
print("Features (X) Head:")
print(X.head())
print("\nTarget (y) Head:")
print(y.head())

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 3. Split Data into Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Shape of Training and Test Sets ---")
print(f"Shape of Training Features (X_train): {X_train.shape}")
print(f"Shape of Training Target (y_train): {y_train.shape}")
print(f"Shape of Test Features (X_test): {X_test.shape}")
print(f"Shape of Test Target (y_test): {y_test.shape}")

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 4. Train the Linear Regression Model ---
model = LinearRegression()
model.fit(X_train, y_train)

print("--- Model Training Complete ---")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 5. Make Predictions ---
y_pred = model.predict(X_test)

print("--- Prediction View ---")
prediction_output_df = pd.DataFrame({
    'Actual Temperature': y_test,
    'Predicted Temperature': y_pred
})
print(prediction_output_df.head())

# Save predictions to CSV
prediction_output_df.to_csv("prediction_output.csv", index=False)
print("Predictions saved to 'prediction_output.csv'.")

print("\n" + "="*50 + "\n") # Separator for clarity

# --- 6. Evaluate the Model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
