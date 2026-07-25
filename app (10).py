import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# LangChain Imports
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Karachi AI Estate Agency", page_icon="🏠", layout="wide")
st.title("🏠 Karachi AI Real Estate & Property Valuer")

# 1. Load Data & Pickle Model
@st.cache_data
def load_dataset():
    return pd.read_csv("/content/property_data.csv")

@st.cache_resource
def load_pickle_model():
    with open("karachi_knn_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data

df = load_dataset()
pickle_data = load_pickle_model()

knn_model = pickle_data["model"]
min_vals = pickle_data["min_vals"]
max_vals = pickle_data["max_vals"]
features = pickle_data["features"]

# 2. Evaluate Model Precision
X = df[features]
y = df["price"]
X_norm = (X - min_vals) / (max_vals - min_vals)
y_pred_all = knn_model.predict(X_norm)

r2 = r2_score(y, y_pred_all)
mae = mean_absolute_error(y, y_pred_all)
rmse = np.sqrt(mean_squared_error(y, y_pred_all))

# 3. Model Accuracy Section
st.subheader("🎯 Model Performance Metrics")
c1, c2, c3 = st.columns(3)
c1.metric("R² Accuracy Score", f"{r2 * 100:.2f}%")
c2.metric("Mean Absolute Error", f"± {mae:.2f} Lakhs")
c3.metric("Root Mean Sq. Error", f"± {rmse:.2f} Lakhs")

# 4. User Interface Inputs
st.sidebar.header("📍 Property Details")
selected_area = st.sidebar.selectbox("Select ", df["location"].unique())
bedrooms = st.sidebar.slider("Bedrooms", int(df["bedroom"].min()), int(df["bedroom"].max()), 3)
bathrooms = st.sidebar.slider("Bath", int(df["bath"].min()), int(df["bath"].max()), 3)
size_sqft = st.sidebar.number_input("area", min_value=500, max_value=10000, value=1800, step=100)

# 5. Prediction Logic using Pickle Model
if st.sidebar.button("Predict Price & Run AI Advisor"):
    # Normalize input using pickled min/max values
    user_raw = pd.DataFrame([[bedroom, bath, area]], columns=features)
    user_norm = (user_raw - min_vals) / (max_vals - min_vals)

    # Predict
    predicted_price = knn_model.predict(user_norm)[0]
    distances, indices = knn_model.kneighbors(user_norm)
    similar_properties = df.iloc[indices[0]]

    st.markdown("---")
    st.subheader("📌 Property Valuation Result")
    st.metric(
        label=f"Valuation for {selected_area}", 
        value=f"PKR {predicted_price:.2f} Lakhs (~ PKR {predicted_price / 100:.2f} Crore)"
    )

    st.write("### 🏠 Top Nearest Comparable Properties")
    st.dataframe(similar_properties[["loaction", "bedroom", "bath", "area", "price"]])

    if api_key:
        try:
            llm = ChatOpenAI(model="gpt-5.4-mini", api_key="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", temperature=0.7)
            prompt = PromptTemplate(
                input_variables=["location", "beds", "baths", "area", "price", "comps"],
                template="""
                You are a senior Karachi Real Estate Broker.
                Client Request: {beds} Bed, {baths} Bath, {sqft} Sq.Ft in {area}, Karachi.
                Estimated Price: PKR {price:.2f} Lakhs.
                Comparables: {comps}
                
                Provide a brief valuation critique and negotiation advice for the Karachi market.
                """
            )
            chain = prompt | llm
            res = chain.invoke({
                "loaction": selected_area, "beds": bedroom, "baths": bathrooms,
                "area": "price": predicted_price,
                "comps": similar_properties[["location", "size", "price"]].to_string()
            })
            st.markdown("### 🤖 AI Agent Report")
            st.markdown(res.content)
        except Exception as e:
            st.error(f"LangChain Error: {e}")
