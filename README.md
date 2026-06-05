# BBRI Stock Price Forecasting Using BiLSTM with Grid Search

This project aims to forecast the stock price of Bank Rakyat Indonesia (BBRI) using a Bidirectional Long Short-Term Memory (BiLSTM) model. Grid Search combined with Time Series Cross-Validation is applied to identify the optimal hyperparameter configuration.

## Project Overview

Stock price forecasting is a challenging time-series problem due to market volatility and nonlinear patterns. This project utilizes a deep learning approach through BiLSTM to capture temporal dependencies in historical stock price data.

The model is optimized using Grid Search and evaluated using Mean Absolute Percentage Error (MAPE) and Mean Squared Error (MSE).

## Dataset

- Company: Bank Rakyat Indonesia (BBRI)
- Variable: Daily Closing Price
- Frequency: Daily
- Period: 2023 - Present

## Methodology

### 1. Data Preprocessing
- Load stock price dataset
- Convert date column to datetime format
- Handle missing values using Forward Fill and Backward Fill
- Generate complete date sequences

### 2. Data Normalization
- Apply MinMaxScaler
- Split dataset into training and testing sets (80:20)

### 3. Time Series Windowing
- Create input sequences using a sliding window approach
- Time Step: 30 days

### 4. BiLSTM Model Development
Model architecture:
- Bidirectional LSTM Layer
- Dropout Layer
- LSTM Layer
- Dropout Layer
- Dense Output Layer

### 5. Hyperparameter Optimization
Grid Search parameters:

| Parameter | Values |
|------------|----------|
| Neurons | 5, 10, 15, 20 |
| Batch Size | 4, 16, 32 |
| Epochs | 50, 100, 150 |
| Dropout | 0.1, 0.2 |
| Time Step | 30 |

### 6. Cross Validation
- TimeSeriesSplit (3 folds)

### 7. Model Evaluation
Metrics:
- Mean Absolute Percentage Error (MAPE)
- Mean Squared Error (MSE)

### 8. Forecasting
- Forecast future stock prices
- Forecast horizon: 90 days

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Results

The optimal BiLSTM model obtained through Grid Search was used for:

- Training prediction
- Testing prediction
- Actual vs Predicted comparison
- 90-day stock price forecasting

## Repository Structure

```text
├── data/
│   └── BBRI.csv
├── notebooks/
├── results/
│   ├── grid_search_results.csv
│   ├── actual_vs_prediction.csv
│   └── forecast_90_days.csv
├── src/
│   └── bilstm_gridsearch.py
├── README.md
└── requirements.txt
```

## Author

Norisa Ayufadhilah
