import os
import subprocess
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

  
# Load Model dan preprocessor
model = joblib.load("linear_model_car_price.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Load CSV
car_data_clean = pd.read_csv("cleaned_carprice.csv")

# Session State Initializer
if "page" not in st.session_state:
    st.session_state.page = "home"
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "history" not in st.session_state:
    st.session_state.history = []

# Page Navigation
st.sidebar.title("Price Prediction")

page_map = {
    "🏎️ Home Page": "home",
    "📝 Input Data": "input",
    "📋 Data Table": "table",
    "📈 Line Chart": "linechart",
    "📊 Bar Chart": "barchart",
    "🧾 History": "history"
}
reverse_map = {v: k for k, v in page_map.items()}

menu = st.sidebar.radio(
    "Select Page:",
    list(page_map.keys()),
    index=list(page_map.values()).index(st.session_state.page)  # sync radio with page
)

st.session_state.page = page_map[menu]

st.sidebar.markdown("---")
st.sidebar.caption("Created by **Atikah DR**")

# Prediction Function
def car_predict(model, preprocessor, data_input):
    try:
        if data_input is None or data_input.empty:
            return None
        
        X_processed = preprocessor.transform(data_input)
        prediction = model.predict(X_processed)

        return float(prediction[0])
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
    
# Home Page
if st.session_state.page == "home":
    st.title("🚗 Car Price Prediction App")
    header_img = "C:/Users/AtikahDR/Documents/Data Science Project/CarPrice/car.jpg"
    st.image(header_img, use_container_width=True)
    st.subheader(" **Welcome!** ")
    st.markdown(
        """ 
        This interactive application helps you estimate the *price of a used car* using a Machine Learning model.
        Fill in the vehicle details (Mileage kmpl, Engine CC, Owner, Accidents, Car Age, Fuel Type, Brand, Transmission,
        Color, Service History, Insurance) and see the estimated price along with the visualisation of the insights generated.
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Why use this application?")
    st.markdown(
         """
            🔍**Fast** — Get price estimates in just a few seconds.  
            📈 **Data-driven** — Models are trained using market data for more accurate predictions.  
            📊 **Visualisation** — See the impact of vehicle features through interactive graphs.  
            🛠️ **Practical** — Suitable for sellers, buyers, analysts, and developers.
            """
        )
    st.subheader("How to use")
    st.markdown(
        """
            1. Select **Input Data** from the sidebar.  
            2. Enter vehicle details (brand, mileage, engine CC, color, transmission, owner, etc.).  
            3. Save your input, open **Data Table** to review and prediction results.  
            4. Open **Line Chart / Bar Chart** to see visualisations and prediction results.
            """  
        )
    st.subheader("Make a Prediction")
    st.write("Select the input menu in the sidebar to enter data and run predictions.")
    if st.button("📝 Input Data"):
        st.session_state.page = "input"
        st.rerun()


# Input Page
if st.session_state.page == "input":
    st.title("🚗 Car Price Prediction by Machine Learning")
    st.write("Fill in the car details bselow to get a price prediction.")

    # Numeric Inputs
    mileage_kmpl = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=35.0, value=20.0, step=0.1)
    engine_cc = st.number_input("Engine CC", min_value=800, max_value=5000, value=2000, step=1)
    owner_count = st.number_input("Owner Count", min_value=1, max_value=5,value=2, step=1)
    accidents_reported = st.number_input("Accidents Reported", min_value=0, max_value=5, value=2, step=1)
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=30, value=5, step=1)

    # Categorical Inputs
    fuel_type =  st.selectbox("Fuel Type", sorted(car_data_clean["fuel_type"].unique()))
    brand = st.selectbox("Brand", sorted(car_data_clean["brand"].unique()))
    transmission = st.selectbox("Transmission", sorted(car_data_clean["transmission"].unique()))
    color = st.selectbox("Color", sorted(car_data_clean["color"].unique()))
    service_history = st.selectbox("Service History", sorted(car_data_clean["service_history"].unique()))
    insurance_valid = st.selectbox("Insurance Report", sorted(car_data_clean["insurance_valid"].unique()))
               
   # Save Button
    if st.button("Save Input Data"):

        # Simpan input ke session state (sebagai DataFrame satu baris)
        st.session_state.input_data = pd.DataFrame([{
            "mileage_kmpl": round(mileage_kmpl, 2),
            "engine_cc": engine_cc,
            "fuel_type": fuel_type,
            "owner_count": owner_count,
            "brand": brand,
            "transmission": transmission,
            "color": color,
            "service_history": service_history,
            "accidents_reported": accidents_reported,
            "insurance_valid": insurance_valid,
            "car_age": car_age
        }])

        # Generate prediction
        prediction = car_predict(model, preprocessor, st.session_state.input_data)
        st.session_state.prediction = prediction

        st.success("✔ Data saved! Go to the TABLE page to continue.")

        # -----------------------------
        # Save to History (Car Price)
        # -----------------------------
        st.session_state.history.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Brand": brand,
            "Mileage (kmpl)": round(mileage_kmpl, 2),
            "Engine (cc)": engine_cc,
            "Fuel": fuel_type,
            "Owner Count": owner_count,
            "Transmission": transmission,
            "Color": color,
            "Service History": service_history,
            "Accidents Reported": accidents_reported,
            "Insurance Valid": insurance_valid,
            "Car Age": car_age,
            "Predicted Price": float(prediction)
        })

        # Pindah ke Table
        st.session_state.page = "table"


# Table Page
elif st.session_state.page == "table":
    st.title("📋 Review Input Data")

    if st.session_state.input_data is None:
        st.warning("⚠ Please enter car data first in INPUT PAGE.")
    else:
        data_t = st.session_state.input_data.T
        data_t.columns = ["Value"]
        st.dataframe(data_t)

        # Prediction Display
        st.subheader("💰 Price Prediction")
        if st.session_state.prediction is not None:
            st.success(f" Estimated Car Price: USD {st.session_state.prediction:,.2f}")
        else:
            st.error("Prediction failed. Please check your inputs.")

# Line Chart Page
elif st.session_state.page =="linechart":
    st.title("📈 Visualisasi Line Chart")

    if st.session_state.input_data is None:
        st.warning("⚠ Please enter car data first in INPUT PAGE.")
    else:
        data_input = st.session_state.input_data

        # add prediction value as another row
        data_input["predicted_price"] = st.session_state.prediction

        # Displaying car data: Brand, Fuel Type, Mileage, and Engine cc
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Brand", data_input["brand"][0])
        with col2:
            st.metric("Fuel Type", data_input["fuel_type"][0])
        with col3:
            st.metric("Mileage Kmpl", data_input["mileage_kmpl"][0])
        with col4:
            st.metric("Engine CC", f"{data_input['engine_cc'][0]:.0f}")

        # Line Chart features
        features = ["owner_count", "car_age", "accidents_reported"]
        values = [data_input[feat][0] for feat in features]

        data_line = pd.DataFrame({
            "Feature": features,
            "Value": values
        })

        st.subheader("📈 Vehicle Feature Line Chart")
        st.line_chart(data_line.set_index("Feature"))

        # Additional car info
        st.subheader("Additional Car Info")

        # Displaying car data: Insurance, Color, Service History, and Transmission 
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Insurance Valid", data_input["insurance_valid"][0])
        with col6:
            st.metric("Color", data_input["color"][0])
        with col7:
            st.metric("Service History", data_input["service_history"][0])
        with col8:
            st.metric("Transmission", data_input["transmission"][0])

        # Prediction Display
        st.subheader("💰 Price Prediction")
        if st.session_state.prediction is not None:
            st.success(f" Estimated Car Price: USD {st.session_state.prediction:,.2f}")
        else:
            st.error("Prediction failed. Please check your inputs.")

# Bar Chart Page
elif st.session_state.page == "barchart":
    st.header("📊 Probability Class Prediction")

    if st.session_state.input_data is None:
        st.warning("⚠ Please enter car data first in INPUT PAGE.")
    else:
        data_input = st.session_state.input_data

        # add prediction value as another row
        data_input["predicted_price"] = st.session_state.prediction

       # Displaying car data: Brand, Fuel Type, Mileage, and Engine cc
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Brand", data_input["brand"][0])
        with col2:
            st.metric("Fuel Type", data_input["fuel_type"][0])
        with col3:
            st.metric("Mileage Kmpl", data_input["mileage_kmpl"][0])
        with col4:
            st.metric("Engine CC", f"{data_input['engine_cc'][0]:.0f}")

        # Bar Chart features
        features = ["owner_count", "car_age", "accidents_reported"]
        values = [int(data_input[feat].iloc[0]) for feat in features]

        data_bar = pd.DataFrame({
            "Feature": features,
            "Value": values
        })
        
        st.subheader("📈 Vehicle Feature Bar Chart")
        st.bar_chart(data_bar.set_index("Feature"), y="Value")

        # Additional car info
        st.subheader("Additional Car Info")

        # Displaying car data: Insurance, Color, Service History, and Transmission 
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Insurance Valid", data_input["insurance_valid"][0])
        with col6:
            st.metric("Color", data_input["color"][0])
        with col7:
            st.metric("Service History", data_input["service_history"][0])
        with col8:
            st.metric("Transmission", data_input["transmission"][0])

        # Prediction Display
        st.subheader("💰 Price Prediction")
        if st.session_state.prediction is not None:
            st.success(f" Estimated Car Price: USD {st.session_state.prediction:,.2f}")
        else:
            st.error("Prediction failed. Please check your inputs.")

# History
# Page 4: History
elif st.session_state.page == "history":
    st.title("🧾 Prediction History")

    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)

        # Styling kolom Predicted Price jadi format currency
        if "Predicted_Price" in history_df.columns:
            history_df["Predicted_Price"] = history_df["Predicted_Price"].apply(
                lambda x: f"${x:,.2f}"
            )

        # Tampilkan history terbaru di paling atas
        st.dataframe(history_df[::-1], use_container_width=True)

    else:
        st.info("Belum ada history prediksi.")

    # Tombol Delete History
    if st.button("🗑️ Delete History"):
        st.session_state.history = []
        st.toast("History deleted successfully!", icon="🗑️")

#  Footer
st.markdown("---")

st.caption("💡 Created by Atikah DR | Machine Learning Prediction Project")
