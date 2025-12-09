import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib

MODEL_PATH = 'forest_reg.pkl'
PIPELINE_PATH = 'housing_full_pipeline.joblib'

@st.cache_resource
def load_model_and_pipeline():
    try:
        model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        return model, pipeline
    except FileNotFoundError:
        st.error(f"⚠️ ML artifact missing. Ensure both '{MODEL_PATH}' and '{PIPELINE_PATH}' are in the directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or pipeline. Error: {e}")
        return None, None

model, pipeline = load_model_and_pipeline()

def create_raw_df_from_features(raw_features):
    data = {
        'longitude': raw_features['Longitude'],
        'latitude': raw_features['Latitude'],
        'housing_median_age': raw_features['Housing Median Age (Years)'],
        'total_rooms': raw_features['Total Rooms'],
        'total_bedrooms': raw_features['Total Bedrooms'],
        'population': raw_features['Population'],
        'households': raw_features['Households'],
        'median_income': raw_features['Median Income (tens of thousands USD)'],
        'ocean_proximity': raw_features['Ocean Proximity']
    }
    return pd.DataFrame([data])

def predict_price(raw_features):
    if model is None or pipeline is None:
        st.warning("Using SIMULATED prediction. Please resolve file loading errors.")
        med_inc = raw_features['Median Income (tens of thousands USD)']
        house_age = raw_features['Housing Median Age (Years)']
        total_rooms = raw_features['Total Rooms']
        ocean_proximity = raw_features['Ocean Proximity']
        base_price = 50000.0
        income_impact = med_inc * 40000.0
        age_impact = (1 / house_age) * 10000.0 if house_age > 0 else 10000
        size_impact = (total_rooms / raw_features['Households']) * 5000.0
        proximity_map = {
            'INLAND': 0.8,
            '<1H OCEAN': 1.2,
            'NEAR BAY': 1.4,
            'NEAR OCEAN': 1.6,
            'ISLAND': 2.0
        }
        location_multiplier = proximity_map.get(ocean_proximity, 1.0)
        predicted_value = (base_price + income_impact + age_impact + size_impact) * location_multiplier
        return min(predicted_value, 500000.0)

    input_df = create_raw_df_from_features(raw_features)
    try:
        processed_data = pipeline.transform(input_df)
    except Exception as e:
        st.error(f"Error during data preprocessing. Error: {e}")
        return 0.0

    return model.predict(processed_data)[0]

st.set_page_config(
    page_title="Housing Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🏡 Automated Housing Price Prediction")
st.markdown("Use the controls below to input the district attributes and get a predicted median house value.")

with st.form(key='prediction_form'):
    st.subheader("Property District Features")
    col1, col2 = st.columns(2)
    with col1:
        med_inc = st.slider('Median Income (in tens of thousands USD)', 0.5, 15.0, 3.5, 0.1)
    with col2:
        house_age = st.slider('Housing Median Age (Years)', 1, 52, 30, 1)

    st.markdown("---")
    st.subheader("District Size and Density")

    col3, col4 = st.columns(2)
    with col3:
        total_rooms = st.number_input('Total Rooms in District', 100, 39000, 2500, 100)
    with col4:
        total_bedrooms = st.number_input('Total Bedrooms in District', 10, 6500, 500, 50)

    col5, col6 = st.columns(2)
    with col5:
        population = st.number_input('Population of District', 30, 35000, 1300, 100)
    with col6:
        households = st.number_input('Total Households in District', 10, 6000, 450, 50)

    st.markdown("---")
    st.subheader("Location Attributes")

    col7, col8 = st.columns(2)
    with col7:
        latitude = st.number_input('Latitude', 32.5, 42.0, 34.05, 0.01, format="%.2f")
    with col8:
        longitude = st.number_input('Longitude', -124.5, -114.0, -118.24, 0.01, format="%.2f")

    ocean_proximity = st.selectbox('Ocean Proximity', ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'], 0)

    st.markdown("---")
    predict_button = st.form_submit_button(label='Predict Median Price', type="primary")

if predict_button:
    raw_input_features = {
        'Median Income (tens of thousands USD)': med_inc,
        'Housing Median Age (Years)': house_age,
        'Total Rooms': total_rooms,
        'Total Bedrooms': total_bedrooms,
        'Population': population,
        'Households': households,
        'Latitude': latitude,
        'Longitude': longitude,
        'Ocean Proximity': ocean_proximity
    }

    with st.spinner('Calculating prediction...'):
        time.sleep(1.5)
        predicted_price = predict_price(raw_input_features)

    predicted_price_formatted = f"${predicted_price:,.0f}"

    st.markdown(
        f"""
        <div style="background-color: #e0f2f1; padding: 25px; border-radius: 12px; border-left: 5px solid #005088;">
            <p style="font-size: 18px; color: #005088; margin: 0;"><strong>Predicted Median House Value:</strong></p>
            <h2 style="font-size: 48px; color: #005088; margin: 5px 0 0 0;">{predicted_price_formatted}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if model is None or pipeline is None:
        st.warning("Prediction is SIMULATED.")
    else:
        st.success("Prediction Complete!")

st.sidebar.title("About the Predictor")
st.sidebar.info(
    "This application uses a trained Machine Learning model (Random Forest) "
    "to estimate the median housing value based on census features."
)
