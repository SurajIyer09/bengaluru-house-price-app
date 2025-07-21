# Updated version of your Streamlit code with:
# - Smaller graph sizes
# - Top 5 affordable locations displayed
# - Cleaned and improved layout

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Streamlit config
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="wide")
st.title("\U0001F3E1 Bengaluru House Price Predictor")

# Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# Clean the data
df.drop(['society', 'availability', 'area_type'], axis=1, inplace=True)
df.dropna(inplace=True)
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)
df.dropna(subset=['bhk'], inplace=True)

# Convert total_sqft to float
def convert_sqft(x):
    try:
        if '-' in x:
            parts = x.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df.dropna(inplace=True)

# Add price per sqft and remove outliers
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
df = df[(df['price_per_sqft'] > 1000) & (df['price_per_sqft'] < 10000)]

# Simplify location
df['location'] = df['location'].apply(lambda x: x.strip())
location_counts = df['location'].value_counts()
df['location'] = df['location'].apply(lambda x: 'other' if location_counts[x] <= 10 else x)

# Encode location
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

# Features and target
X = df[['total_sqft', 'bath', 'balcony', 'bhk', 'location']]
y = np.log1p(df['price'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

# Sidebar - User input
st.sidebar.header("\U0001F3E0 Enter House Details")
sqft = st.sidebar.number_input("Total Square Feet", 300, 10000, step=50, value=1000)
bath = st.sidebar.slider("Bathrooms", 1, 5, 2)
bhk = st.sidebar.slider("BHK", 1, 5, 2)
balcony = st.sidebar.slider("Balconies", 0, 5, 1)
location_names = le.classes_
selected_location = st.sidebar.selectbox("Location", sorted(location_names))
encoded_location = le.transform([selected_location])[0]

# Make prediction
input_data = pd.DataFrame([[sqft, bath, balcony, bhk, encoded_location]],
                          columns=['total_sqft', 'bath', 'balcony', 'bhk', 'location'])
predicted_log_price = model.predict(input_data)[0]
predicted_price = np.expm1(predicted_log_price)

# Display result
st.sidebar.markdown(f"### \U0001F4B0 Estimated Price: ₹{predicted_price:.2f} Lakhs")

# Graphs section
st.subheader("\U0001F4CA Price Distribution")
fig1, ax1 = plt.subplots(figsize=(6, 3))
sns.histplot(df['price'], bins=30, kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("\U0001F4C8 Price vs Total Sqft")
fig2, ax2 = plt.subplots(figsize=(6, 3))
sns.scatterplot(data=df, x='total_sqft', y='price', hue='bhk', palette='viridis', alpha=0.6, ax=ax2)
st.pyplot(fig2)

# Top 5 affordable areas
st.subheader("\U0001F4C5 Top 5 Most Affordable Areas (Avg Price per Sqft)")
reverse_encoded_locations = dict(zip(le.transform(le.classes_), le.classes_))
df['decoded_location'] = df['location'].map(reverse_encoded_locations)
mean_pps = df.groupby('decoded_location')['price_per_sqft'].mean().sort_values().head(5)
st.table(mean_pps.reset_index().rename(columns={'decoded_location': 'Location', 'price_per_sqft': 'Avg Price/Sqft'}))

# Model Performance
st.markdown(f"**\U0001F4CC Model R² Score:** {r2:.2f} &nbsp;&nbsp; | &nbsp;&nbsp; **RMSE:** {rmse:.2f} Lakhs")
