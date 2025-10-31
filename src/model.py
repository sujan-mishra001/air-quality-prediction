import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense ,Reshape 
import os


def create_sequence (data,seq_length,forecast_len):
    X,y=[],[]
    for i in range(len(data)-seq_length-forecast_len+1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+forecast_len])
    return np.array(X), np.array(y)

def model_lstm(X,y,seq_length,num_features,forecast_length,st):
  
    if st: 
        model=Sequential()
        model.add(LSTM(128,input_shape=(seq_length,num_features)))

        model.add(Dense(forecast_length* num_features))
        model.add(Reshape((forecast_length,num_features)))
        model.compile(optimizer='adam',loss='mse')
        model.fit(
                    X,y,
                    epochs=5,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                    )
    return model 
