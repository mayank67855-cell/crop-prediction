import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load model and encoder
# -------------------------------
with open("crop_model.pkl", "rb") as f:
    model, le = pickle.load(f)

# Load dataset (for visualization)
df = pd.read_csv("Crop_recommendation (1).csv")
X = df.drop("label", axis=1)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌱 Crop Recommendation System")
st.write("Enter soil and climate conditions to get the best crop suggestion.")

# Sidebar inputs
N = st.sidebar.slider("Nitrogen (N)", 0, 150, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 150, 50)
K = st.sidebar.slider("Potassium (K)", 0, 150, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# Prediction
if st.sidebar.button("Recommend Crop"):
    input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(input_data)
    crop = le.inverse_transform(prediction)[0]
    st.success(f"🌾 Recommended Crop: **{crop}**")

# -------------------------------
# Data Visualization
# -------------------------------
st.subheader("📊 Crop Distribution in Dataset")
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x=df["label"], order=df["label"].value_counts().index, ax=ax)
plt.xticks(rotation=90)
st.pyplot(fig)

st.subheader("🔍 Feature Importance")
importance = model.feature_importances_
feature_imp = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
st.bar_chart(feature_imp.set_index("Feature"))
