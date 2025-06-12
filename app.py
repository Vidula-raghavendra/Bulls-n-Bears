import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from flask import Flask, request, render_template

# New imports for plotting and news
import plotly.graph_objects as go
from GoogleNews import GoogleNews

# Keras/TensorFlow imports
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

app = Flask(__name__)

# ------------------------ News Function ------------------------
def get_news(query, limit=5):
    """Fetches recent news articles for a given query."""
    try:
        googlenews = GoogleNews(lang='en', region='US')
        googlenews.search(f"{query} stock")
        results = googlenews.result(sort=True)[:limit] # sort=True gets most recent
        articles = []
        for r in results:
            title = r.get("title")
            link = r.get("link")
            # Ensure the link is fully qualified
            if title and link and not link.startswith('http'):
                link = f"https://news.google.com{link[1:]}"
            if title and link:
                articles.append({"title": title, "link": link})
        return articles
    except Exception as e:
        app.logger.error(f"Error fetching news for {query}: {e}")
        return [] # Return empty list on error

# ------------------------ LSTM Prediction Function ------------------------
# NOTE: This function trains a model from scratch on every request. This is highly
# inefficient and will cause long wait times.
def create_lstm_model_and_predict(data, future_days=7):
    """
    Creates, trains, and uses an LSTM model to predict future stock prices.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)])

    future_predictions = []
    last_sequence = scaled_data[-prediction_days:]

    for _ in range(future_days):
        input_seq = np.reshape(last_sequence, (1, prediction_days, 1))
        predicted_scaled = model.predict(input_seq, verbose=0)[0][0]
        future_predictions.append(predicted_scaled)
        last_sequence = np.append(last_sequence[1:], [[predicted_scaled]], axis=0)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    return future_prices

# ------------------------ Web App Routes ------------------------
@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and displays prediction results."""
    ticker = request.form.get('ticker')
    if not ticker:
        return render_template('results.html', error="Ticker symbol cannot be empty.")

    try:
        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=365 * 2)
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)

        if data.empty:
            error_msg = f"No data found for ticker '{ticker}'. Please check the symbol (e.g., 'AAPL', 'GOOG', 'TCS.NS')."
            return render_template('results.html', ticker=ticker, error=error_msg)

        company_name = stock.info.get('longName', ticker.upper())
        
        # *** NEW: Fetch news articles ***
        news_articles = get_news(company_name)

        last_10_days_df = data.tail(10).reset_index()
        last_10_days_df['Date'] = last_10_days_df['Date'].dt.strftime('%Y-%m-%d')
        last_10_days = last_10_days_df[['Date', 'Close']].to_dict('records')
        last_10_days = [{'index': row['Date'], 'Close': row['Close']} for row in last_10_days]

        future_prices = create_lstm_model_and_predict(data)
        future_dates = [(end + datetime.timedelta(days=i + 1)) for i in range(len(future_prices))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Predicted Close', line=dict(dash='dash')))
        fig.update_layout(
            title=f'Stock Price Prediction for {company_name}',
            xaxis_title='Date',
            yaxis_title='Stock Price (in stock\'s currency)',
            template='plotly_white'
        )
        plot_html = fig.to_html(full_html=False)

        # *** NEW: Pass news_articles to the template ***
        return render_template('results.html',
                               ticker=ticker,
                               company_name=company_name,
                               plot_html=plot_html,
                               last_10_days=last_10_days,
                               news_articles=news_articles)

    except Exception as e:
        app.logger.error(f"Error processing ticker {ticker}: {e}")
        return render_template('results.html', ticker=ticker, error=f"An error occurred: {str(e)}")

# ------------------------ Run Server ------------------------
if __name__ == '__main__':
    app.run(debug=True)