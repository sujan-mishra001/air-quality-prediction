import streamlit as st
import pandas as pd
from datetime import date, timedelta,datetime
from data import fetch_data,preprocess_data,convert_dataframe
from model import create_sequence,model_lstm
st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/258149/pexels-photo-258149.jpeg?cs=srgb&dl=pexels-pixabay-258149.jpg&fm=jpg");
            background-size: cover;
            background-attachment: centered;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
st.title("Air Quality by OpenWeather API")

# fetching data in json format from api
col1,col2=st.columns(2)
with col1:
    lat=st.number_input("lattitude")
with col2:
    lon=st.number_input("longitude")

default_end =  date.today() 
col1,col2=st.columns(2)
with col1:
    e=st.date_input("End Date",value=default_end)

with col2:

    s=st.date_input("Start Date",value=e- timedelta(days=107))

start = int(datetime.combine(s, datetime.min.time()).timestamp())
end = int(datetime.combine(e, datetime.min.time()).timestamp())
    

if st.button("Find Data"):
    
    data=fetch_data(lat,lon,start,end)
    df=convert_dataframe(data)
    st.subheader(f"The Weather data from {pd.to_datetime(start,unit='s').date()} to {pd.to_datetime(end,unit='s').date()} of lattitude: {lat} & longitude: {lon} are:")
    st.write(df)


    datas,features,scaler=preprocess_data(df)
    seq_length=360
    forecast_length=24
    X,y=create_sequence(datas,seq_length,forecast_length)

    num_features=len(features)

    model=model_lstm(X,y,seq_length,num_features,forecast_length,st)
    
    last_seq = X[-1].reshape(1, seq_length, num_features)
    forecast = model.predict(last_seq)

    forecast = forecast.reshape(forecast_length, num_features)
    forecast = scaler.inverse_transform(forecast)

    value=pd.DataFrame(forecast,columns=features)
    value['time'] =pd.date_range("00:00", "23:00", freq="h",inclusive="both").time
    value=value.set_index('time')
    st.session_state['value']=value

if st.button("Forecast"):
    value=st.session_state['value']
    st.header(f"24-Hour Weather Forecast {pd.to_datetime(end,unit='s').date()+timedelta(days=1)}")
    st.write(value)


    



   
    



