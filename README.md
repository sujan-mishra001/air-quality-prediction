# Air Quality Prediction

This project predicts air quality using historical data from the OpenWeather API and displays results in a Streamlit web app.

## Features

- Fetches air pollution data for a given latitude and longitude.
- Preprocesses and normalizes data for model training.
- Trains an LSTM neural network to forecast air quality metrics.
- Displays historical and forecasted air quality in an interactive dashboard.

## Requirements

See [requirements.txt](requirements.txt) for dependencies.

## Setup

1. Create and activate a virtual environment:
    ```sh
    python -m venv myvenv
    source myvenv/Scripts/activate  
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:
```sh
streamlit run src/app.py
```

## File Structure

- (src/app.py): Streamlit app entry point.
- (src/data.py): Data fetching and preprocessing.
- (src/model.py): LSTM model creation and training.

## Notes

- You need an internet connection to fetch data from OpenWeather API.
- The API key is hardcoded in (src/data.py); replace it with your own.
