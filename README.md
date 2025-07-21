# 🏠 Bengaluru House Price Predictor

A Machine Learning-powered web application built using **Streamlit** that predicts house prices in **Bengaluru** based on user inputs like area, number of bedrooms, bathrooms, and location. Powered by **Random Forest Regressor**, the app is trained on real-world Bengaluru housing data.

🔗[ **Live App**: [Click to Launch](https://bengaluru-house-price-app.streamlit.app/)
🔗 **LinkedIn**: [Suraj Iyer](https://www.linkedin.com/in/suraj-iyer-805599266/)

---

## 🚀 Features

- 🎯 Predict house prices in Bengaluru (Lakh ₹)
- 📍 Input parameters: Location, Area (sqft), BHK, Bathrooms
- 📊 Top 5 most affordable localities shown dynamically
- 🔍 Clean and responsive UI with Streamlit sidebar
- 🧠 Trained with RandomForestRegressor
- ☁️ Deployed on Streamlit Community Cloud

---

## 🧠 Machine Learning Details

- **Model**: RandomForestRegressor (from `sklearn`)
- **Training Data**: Real Bengaluru house price dataset
- **Target Variable**: Price per square foot (converted to Lakhs ₹)

## 📁 Project Structure

```
.
├── app.py                   # Streamlit frontend
├── model.py                 # Data processing & model training
├── model.pkl                # Trained model
├── Bengaluru_House_Data.csv # Dataset
├── requirements.txt         # Required Python packages
├── README.md                # Project documentation
├── .gitignore               # Files to ignore in version control
└── screenshot.png           # App screenshot
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bengaluru-house-price-app.git
cd bengaluru-house-price-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app locally

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push your project to GitHub
2. Visit: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in and click “New app”
4. Connect your repo and branch
5. Set the main file as `app.py`
6. Hit “Deploy”

---

## 📝 Sample Prediction

| Area (sqft) | BHK | Bath | Location     | Predicted Price (Lakh ₹) |
|-------------|-----|------|--------------|---------------------------|
| 1000        | 2   | 2    | Whitefield   | ₹45.82                    |
|             |     |      |              |                           |

---

## 📌 Screenshot
<img width="1885" height="910" alt="image" src="https://github.com/user-attachments/assets/b263cc17-ba95-41bf-97a0-cd300f5fa449" />




---

## 🙌 Acknowledgements

- Streamlit for rapid UI development
- scikit-learn for machine learning
- Kaggle for providing the dataset

---

## 📨 Contact

Created by **Suraj Iyer**  
📧 surajiyer0912@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/suraj-iyer-805599266/)

---





