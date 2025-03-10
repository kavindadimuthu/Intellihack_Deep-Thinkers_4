# Stock Price Prediction Model

## Overview
This project develops a machine learning model to predict a stock's closing price 5 trading days into the future using historical stock price data. The solution leverages an LSTM (Long Short-Term Memory) neural network to capture temporal patterns in the data, performs comprehensive exploratory data analysis (EDA), and evaluates the model using both statistical metrics and simulated trading performance. The workflow is documented in a Jupyter notebook, adhering to the requirements of the Stock Price Prediction Challenge.

## Table of Contents
1. [Project Description](#project-description)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Usage](#usage)
7. [Results](#results)
8. [Limitations](#limitations)
9. [Future Improvements](#future-improvements)

## Project Description
The goal of this project is to predict the closing price of a stock 5 trading days ahead using historical data (`question4-stock-data.csv`). The solution includes:
- Exploratory Data Analysis (EDA) to uncover trends, anomalies, and patterns.
- Feature engineering to enhance predictive power.
- Development and training of an LSTM-based model.
- Evaluation using statistical metrics (MAE, RMSE, directional accuracy) and a simulated trading strategy.
- Documentation of the approach, findings, and limitations.

This project meets the evaluation criteria:
- **Exploratory Data Analysis (50%)**: Detailed visualizations and insights.
- **Prediction Accuracy (10%)**: Assessed via RMSE and directional accuracy.
- **Documentation and Insights (30%)**: Clear explanations in the notebook.
- **Limitation Analysis and Improvement Strategies (20%)**: Identified limitations and proposed enhancements.

## Requirements
To run the project, ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `sklearn`
- `talib`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/kavindadimuthu/Intellihack_Deep-Thinkers_4.git
   cd stock-price-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Create a `requirements.txt` file by listing the libraries above if not already provided.)*
3. Ensure the dataset `question4-stock-data.csv` is in the project directory.

## Dataset
The dataset (`question4-stock-data.csv`) contains historical stock price data with the following columns:
- `Date`: Trading date (1980-03-17 to present).
- `Adj Close`: Adjusted closing price.
- `Close`: Closing price.
- `High`: Highest price of the day.
- `Low`: Lowest price of the day.
- `Open`: Opening price.
- `Volume`: Trading volume.

The dataset spans 11,291 entries, with some missing values handled during preprocessing.

## Methodology
### 1. Exploratory Data Analysis (EDA)
- **Data Inspection**: Checked for missing values, data types, and basic statistics (`df.info()`, `df.describe()`).
- **Missing Values**: Identified and quantified (e.g., 145 missing in `Volume`).
- **Visualizations**: Plotted actual vs. predicted prices, training/validation loss, and cumulative trading returns.

### 2. Data Preprocessing
- Dropped unnecessary column (`Unnamed: 0`).
- Converted `Date` to datetime and sorted chronologically.
- Handled missing values with forward-fill (`ffill`) and backward-fill (`bfill`).
- Replaced zero `Open` prices with the previous day's `Close`.
- Removed rows with zero volume and no price change (assumed holidays).
- Filled remaining zero-volume rows with forward-fill when prices changed.

### 3. Feature Engineering
- Created a target variable: `Target_Close_5d` (closing price 5 days ahead).
- Scaled features using `MinMaxScaler` for LSTM compatibility.
- Prepared sequences with a 20-day time step for temporal modeling.

### 4. Model Development
- **Model Choice**: LSTM neural network with dropout and L2 regularization.
- **Architecture**:
  - LSTM layer (50 units).
  - Dropout (0.2).
  - Dense layers with L2 regularization.
  - Optimizer: `Adamax`.
- **Training**: 100 epochs with a learning rate reduction callback.

### 5. Evaluation
- **Statistical Metrics**:
  - Mean Absolute Error (MAE).
  - Root Mean Squared Error (RMSE).
  - Directional Accuracy (% of correct price movement predictions).
- **Trading Simulation**: 
  - Strategy: Buy if predicted price increases >1%, sell if decreases >1%.
  - Metric: Total cumulative return.

## Usage
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook stock_price_prediction.ipynb
   ```
2. Run all cells to:
   - Load and preprocess the data.
   - Train the LSTM model.
   - Generate predictions.
   - Visualize results.
   - Save predictions to `predictions.csv`.

## Results
- **Model Performance** :
  - MAE: 24.1062.
  - RMSE: 29.3561.
  - Directional Accuracy: 50.57%.
- **Trading Performance**: Total Return: -64.17% (indicating a loss, suggesting model refinement needed).
- **Visualizations**:
  - Actual vs. Predicted Prices.
  - Training/Validation Loss Over Epochs.
  - Cumulative Returns from Trading Strategy.

The predictions are saved in `predictions.csv` with columns: `Date`, `Close`, `Predicted_Close_5d`, `Target_Close_5d`.

## Limitations
1. **Data Quality**: Missing values and zero `Open` prices may introduce bias despite preprocessing.
2. **Model Overfitting**: Negative trading returns suggest the model struggles to generalize to unseen data.
3. **Feature Set**: Limited to price and volume; external factors (e.g., news, market sentiment) are not included.
4. **Trading Strategy**: Simple threshold-based strategy (±1%) may not capture optimal trading opportunities.
5. **Time Horizon**: Predicting 5 days ahead is challenging due to market volatility and noise.

## Future Improvements
1. **Additional Data**: Incorporate macroeconomic indicators, news sentiment, or technical indicators (e.g., RSI, MACD via `talib`).
2. **Model Tuning**: Experiment with deeper LSTM architectures, GRU layers, or ensemble methods.
3. **Hyperparameter Optimization**: Use grid search or Bayesian optimization for learning rate, units, etc.
4. **Advanced Trading Strategy**: Implement stop-loss, take-profit, or reinforcement learning-based trading.
5. **Cross-Validation**: Use time-series cross-validation to better assess generalization.
