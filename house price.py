# ================================
# 🏠 House Price Prediction Web App
# ================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page title
st.title("🏠 House Price Prediction App")

# Random seed
np.random.seed(42)

# ================================
# 1️⃣ Dummy Dataset Create karna
# ================================

data_size = 500

data = pd.DataFrame({
    'SqFt': np.random.randint(500, 3000, data_size),
    'Bedrooms': np.random.randint(1, 6, data_size),
    'Bathrooms': np.random.randint(1, 4, data_size),
    'Age': np.random.randint(0, 30, data_size),
    'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], data_size)
})

# Price calculation
data['Price'] = (
    data['SqFt'] * 150 +
    data['Bedrooms'] * 10000 +
    data['Bathrooms'] * 5000 -
    data['Age'] * 2000 +
    np.random.randint(-20000, 20000, data_size)
)

# ================================
# 2️⃣ Preprocessing
# ================================

le = LabelEncoder()
data['Location'] = le.fit_transform(data['Location'])

X = data.drop("Price", axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 3️⃣ Model Training
# ================================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================================
# 4️⃣ Sidebar User Input
# ================================

st.sidebar.header("Enter House Details")

sqft = st.sidebar.slider("Square Feet", 500, 3000, 1000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 4, 2)
age = st.sidebar.slider("Age of House", 0, 30, 5)
location = st.sidebar.selectbox("Location", ['Urban', 'Suburban', 'Rural'])

location_encoded = le.transform([location])[0]

user_input = pd.DataFrame([[sqft, bedrooms, bathrooms, age, location_encoded]],
                          columns=X.columns)

# ================================
# 5️⃣ Prediction Button
# ================================

if st.button("Predict Price"):
    prediction = model.predict(user_input)
    st.success(f"🏷️ Predicted House Price: ₹ {round(prediction[0], 2)}")

# ================================
# 6️⃣ Model Evaluation Display
# ================================

st.subheader("📊 Model Evaluation")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**MAE:** {round(mae,2)}")
st.write(f"**MSE:** {round(mse,2)}")
st.write(f"**R² Score:** {round(r2,2)}")
