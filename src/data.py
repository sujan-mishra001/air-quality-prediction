import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fetch_data(lat,lon,start,end):
    try:
        api="a839ea7b554afb191fc13841adde72b2"

        url=f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api}"
        response=requests.get(url)
        response=response.json()

        return response

    except requests.exceptions.RequestException as e:
        print(f"fetch error :{e}")
        return None
    
def convert_dataframe(data):
    df=pd.json_normalize(data['list'])
    df['dt']=pd.to_datetime(df['dt'],unit='s')
    df.columns=df.columns.str.replace("components.","").str.replace("main.","")
    return df

    
def preprocess_data(raw_data):
    features=["aqi"	,"co",	"no",	"no2",	"o3",	"so2",	"pm2_5"	,"pm10","nh3"]
    data=raw_data[features]
    scaler=MinMaxScaler()
    df_scaled=scaler.fit_transform(data)
    return df_scaled,features,scaler
