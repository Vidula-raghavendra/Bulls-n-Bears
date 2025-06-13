import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from flask import Flask, request, render_template

# Imports for plotting and news
import plotly.graph_objects as go
from GoogleNews import GoogleNews

# Keras/TensorFlow imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

app = Flask(__name__)

# --- REAL LSTM MODEL TRAINING FUNCTION ---
def create_and_train_lstm(data):
    """
    Creates, trains, and returns an LSTM model, along with the scaler and key parameters.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60  # Lookback period

    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM Model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model. verbose=0 keeps the console clean during requests.
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    return model, scaler, scaled_data, prediction_days

# --- NEWS FUNCTION ---
def get_news(query, limit=5):
    """Fetches recent news articles for a given query."""
    try:
        googlenews = GoogleNews(lang='en', region='US')
        googlenews.search(f"{query} stock")
        results = googlenews.result(sort=True)[:limit]
        articles = []
        for r in results:
            title = r.get("title")
            link = r.get("link")
            if title and link and not link.startswith('http'):
                link = f"https://news.google.com{link[1:]}"
            if title and link:
                articles.append({"title": title, "link": link})
        return articles
    except Exception as e:
        app.logger.error(f"Error fetching news for {query}: {e}")
        return []

# --- WEB APP ROUTES ---
@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the backtesting request and displays accuracy results."""
    ticker = request.form.get('ticker')
    backtest_date_str = request.form.get('backtest_date')

    if not ticker or not backtest_date_str:
        return render_template('results.html', error="Ticker and a valid backtest date are required.")

    try:
        backtest_date = datetime.datetime.strptime(backtest_date_str, '%Y-%m-%d')
        
        # --- 1. Data Fetching ---
        training_end_date = backtest_date
        training_start_date = training_end_date - datetime.timedelta(days=365*2)
        actual_data_end_date = training_end_date + datetime.timedelta(days=12) # Fetch extra days to ensure we get 7 trading days
        
        stock = yf.Ticker(ticker)
        training_data = stock.history(start=training_start_date, end=training_end_date)
        actual_data = stock.history(start=training_end_date, end=actual_data_end_date)

        if training_data.empty or len(training_data) < 60:
            return render_template('results.html', error=f"Not enough historical data for {ticker} before {backtest_date_str}. Please choose an earlier date.")

        # --- 2. Model Training and Prediction ---
        model, scaler, scaled_data, prediction_days = create_and_train_lstm(training_data)
        
        last_sequence = scaled_data[-prediction_days:]
        future_predictions_scaled = []
        for _ in range(7):
            input_seq = np.reshape(last_sequence, (1, prediction_days, 1))
            predicted_scaled = model.predict(input_seq, verbose=0)[0][0]
            future_predictions_scaled.append(predicted_scaled)
            last_sequence = np.append(last_sequence[1:], [[predicted_scaled]], axis=0)
        
        predicted_prices = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()

        # --- 3. Accuracy Calculation ---
        actual_prices = actual_data['Close'].values[1:8] # Skip the test date, take next 7
        
        if len(actual_prices) < 7:
            return render_template('results.html', error=f"Could not retrieve 7 full trading days after {backtest_date_str} (market might have been closed). Please select a different date.")
            
        predicted_prices = predicted_prices[:len(actual_prices)]

        mae = mean_absolute_error(actual_prices, predicted_prices)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        prediction_dates = pd.to_datetime(actual_data.index[1:8])
        comparison_data = pd.DataFrame({
            'Date': prediction_dates.strftime('%Y-%m-%d'),
            'Actual Price': actual_prices,
            'Predicted Price': predicted_prices
        }).to_dict('records')

        # --- 4. Visualization ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=training_data.index, y=training_data['Close'], mode='lines', name='Historical Price', line=dict(color='#4299e1')))
        fig.add_trace(go.Scatter(x=prediction_dates, y=actual_prices, mode='lines', name='Actual Price', line=dict(color='#2ecc71', width=3)))
        fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines', name='Predicted Price', line=dict(color='#e74c3c', dash='dash', width=3)))
        
        fig.update_layout(
            title=f"Backtest Results for {ticker} from {backtest_date_str}",
            xaxis_title=None, yaxis_title='Stock Price', template='plotly_white',
            legend=dict(x=0.01, y=0.99, bordercolor='lightgray', borderwidth=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        plot_html = fig.to_html(full_html=False, config={'displayModeBar': False})

        # --- 5. Other Info ---
        company_info = stock.info
        company_name = company_info.get('longName', ticker.upper())
        news_articles = get_news(company_name)

        return render_template('results.html',
                               ticker=ticker,
                               company_name=company_name,
                               plot_html=plot_html,
                               mae=f"${mae:.2f}",
                               mape=f"{mape:.2f}%",
                               comparison_data=comparison_data,
                               news_articles=news_articles)

    except Exception as e:
        app.logger.error(f"Major error processing request: {e}")
        return render_template('results.html', error="An unexpected error occurred. Please check your inputs or try again later.")

# ------------------------ RUN SERVER ------------------------
if __name__ == '__main__':
    app.run(debug=True)