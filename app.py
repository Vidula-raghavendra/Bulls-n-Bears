import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from urllib.parse import urljoin
import traceback

# --- ML & Plotting Imports ---
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import plotly.graph_objects as go

# --- New News Import ---
from GoogleNews import GoogleNews  # pip install GoogleNews

# --- App Initialization ---
app = Flask(__name__)

# --- Helper Functions ---

def get_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return history, stock.info

def get_news(query, limit=5):
    googlenews = GoogleNews(lang='en', region='US')
    googlenews.search(f"{query} stock")
    results = googlenews.result()[:limit]
    articles = []
    for r in results:
        title = r.get("title")
        link = r.get("link")
        if title and link:
            articles.append({"title": title, "link": link})
    return articles

def get_index_info(info):
    sector = info.get('sector', 'N/A')
    industry = info.get('industry', 'N/A')
    return f"{sector} / {industry}" if sector != 'N/A' else "N/A"

def create_lstm_model_and_predict(data, future_days=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=3)])

    # Predict future values
    future_predictions = []
    last_sequence = scaled_data[-prediction_days:]

    for _ in range(future_days):
        input_seq = np.reshape(last_sequence, (1, prediction_days, 1))
        predicted_scaled = model.predict(input_seq, verbose=0)[0][0]
        future_predictions.append(predicted_scaled)
        last_sequence = np.append(last_sequence[1:], [[predicted_scaled]], axis=0)

    # Inverse transform predictions
    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    return future_prices


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    try:
        data, info = get_stock_data(ticker)
        if data.empty:
            return render_template('results.html', error=f"No data for '{ticker}'", ticker=ticker)

        last10 = data[['Close']].tail(10).iloc[::-1]
        last10.index = last10.index.strftime('%Y-%m-%d')
        last10['Close'] = last10['Close'].round(2)

        company = info.get('longName', ticker)
        stock_news = get_news(company)
        index_name = get_index_info(info)
        index_news = get_news(index_name) if index_name != "N/A" else []

        # ⚡️ Use CDN version of Plotly JS
        future_preds = create_lstm_model_and_predict(data, 7)
        next_dates = pd.to_datetime([data.index[-1] + pd.DateOffset(days=i) for i in range(1, len(future_preds)+1)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=next_dates, y=future_preds, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"{company} ({ticker}) Price & Prediction", template='plotly_white')
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return render_template('results.html',
                               ticker=ticker,
                               company_name=company,
                               plot_html=plot_html,
                               last_10_days=last10.to_dict('records'),
                               stock_news=stock_news,
                               index_name=index_name,
                               index_news=index_news)
    except Exception as e:
        traceback.print_exc()
        return render_template('results.html', error=f"Critical error: {e}", ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True)
