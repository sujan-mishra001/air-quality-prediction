import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime
import plotly.express as px
from data import fetch_data, preprocess_data, convert_dataframe
from model import create_sequence, model_lstm

st.set_page_config(page_title="Air Quality Forecast", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/258149/pexels-photo-258149.jpeg?cs=srgb&dl=pexels-pixabay-258149.jpg&fm=jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    </style>
""", unsafe_allow_html=True)

st.title("☢   Air Quality Forecast using OpenWeather API")

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input(" Latitude", value=27.7, format="%.4f")
with col2:
    lon = st.number_input(" Longitude", value=85.3, format="%.4f")

default_end = date.today()
col1, col2 = st.columns(2)
with col1:
    end_date = st.date_input("End Date", value=default_end)
with col2:
    start_date = st.date_input(" Start Date", value=end_date - timedelta(days=107))

start = int(datetime.combine(start_date, datetime.min.time()).timestamp())
end = int(datetime.combine(end_date, datetime.min.time()).timestamp())

features = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

if st.button("Fetch & Train Model"):
    with st.spinner("Fetching data and training model..."):
        data = fetch_data(lat, lon, start, end)
        df = convert_dataframe(data)
        st.session_state['df'] = df

        datas, feat, scaler = preprocess_data(df)
        seq_length, forecast_length = 360, 24
        X, y = create_sequence(datas, seq_length, forecast_length)
        num_features = len(feat)

        model = model_lstm(X, y, seq_length, num_features, forecast_length, st)
        last_seq = X[-1].reshape(1, seq_length, num_features)

        forecast = model.predict(last_seq).reshape(forecast_length, num_features)
        forecast = scaler.inverse_transform(forecast)

        forecast_df = pd.DataFrame(forecast, columns=feat)
        forecast_df['time'] = pd.date_range("00:00", "23:00", freq="h").time
        forecast_df = forecast_df.set_index('time')

        st.session_state['forecast_df'] = forecast_df

    st.success(" Data fetched and model trained successfully.")

if 'df' in st.session_state and 'forecast_df' in st.session_state:
    tab1, tab2 = st.tabs([" Total Fetched Data ", "24-Hour Forecast"])

    with tab1:
        df = st.session_state['df']
        st.subheader(f"Weather Data from {start_date} to {end_date-timedelta(days=1)} for ({lat}, {lon})")
        st.dataframe(df)

        selected = st.multiselect("Select features to visualize", features, default=["aqi", "pm2_5"])
        if selected:
            for feat in selected:
                fig = px.line(df, x=df.index, y=feat, title=f"{feat.upper()} Over Time", markers=True)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        forecast_df = st.session_state['forecast_df']
        forecast_date = pd.to_datetime(end, unit='s').date() + timedelta(days=1)
        st.subheader(f"24-Hour Forecast of {forecast_date} for ({lat}, {lon})")

        st.dataframe(forecast_df, use_container_width=True)

        selected = st.multiselect("Select features to visualize", features, default=["aqi", "pm2_5", "pm10"])
        if selected:
            for feat in selected:
                fig = px.line(forecast_df, x=forecast_df.index, y=feat,
                              title=f"Forecasted {feat.upper()} Levels",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)
