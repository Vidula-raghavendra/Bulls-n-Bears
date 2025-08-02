import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from flask import Flask, request, render_template

# Plotting and news
import plotly.graph_objects as go
from GoogleNews import GoogleNews

# Load pre-trained model
from keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load model and scaler (trained locally once)
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")
prediction_days = 90  # Must match what was used during training


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


# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker')
    backtest_date_str = request.form.get('backtest_date')

    if not ticker or not backtest_date_str:
        return render_template('results.html', error="Ticker and a valid date are required.")

    try:
        backtest_date = datetime.datetime.strptime(backtest_date_str, '%Y-%m-%d').date()
        today = datetime.date.today()
        is_future_forecast = (today - backtest_date).days <= 1

        training_end_date = datetime.datetime.combine(backtest_date, datetime.datetime.min.time())
        training_start_date = training_end_date - datetime.timedelta(days=365*3)

        stock = yf.Ticker(ticker)
        training_data = stock.history(start=training_start_date, end=training_end_date)

        if training_data.empty or len(training_data) < prediction_days:
            return render_template('results.html', error=f"Not enough historical data for {ticker} before {backtest_date_str}.")

        # Prepare data
        scaled_data = scaler.transform(training_data['Close'].values.reshape(-1, 1))
        last_sequence = scaled_data[-prediction_days:]
        future_predictions_scaled = []

        for _ in range(7):
            input_seq = np.reshape(last_sequence, (1, prediction_days, 1))
            predicted_scaled = model.predict(input_seq, verbose=0)[0][0]
            future_predictions_scaled.append(predicted_scaled)
            last_sequence = np.append(last_sequence[1:], [[predicted_scaled]], axis=0)

        predicted_prices_smooth = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1)).flatten()

        # Add slight noise for realism
        avg_price = training_data['Close'].mean()
        noise_level = avg_price * 0.0075
        noise = np.random.normal(0, noise_level, predicted_prices_smooth.shape)
        predicted_prices = predicted_prices_smooth + noise

        if is_future_forecast:
            mode = "Forecast"
            prediction_dates = pd.to_datetime([training_end_date + datetime.timedelta(days=i) for i in range(1, 8)])
            comparison_data = [{'Date': date.strftime('%Y-%m-%d'), 'Predicted Price': price, 'Actual Price': 'N/A'}
                               for date, price in zip(prediction_dates, predicted_prices)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=training_data.index, y=training_data['Close'], mode='lines',
                                     name='Historical Price', line=dict(color='#4299e1')))
            fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines',
                                     name='Forecasted Price', line=dict(color='#e74c3c', width=3)))
            fig.update_layout(title=f"7-Day Price Forecast for {ticker} from {backtest_date_str}")
        else:
            mode = "Backtest"
            actual_start = training_end_date + datetime.timedelta(days=1)
            actual_end = actual_start + datetime.timedelta(days=14)
            actual_data = stock.history(start=actual_start, end=actual_end)

            if len(actual_data) < 7:
                return render_template('results.html', error=f"No 7 full trading days after {backtest_date_str}.")

            actual_prices = actual_data['Close'].values[:7]
            prediction_dates = pd.to_datetime(actual_data.index[:7])
            predicted_prices = predicted_prices[:len(actual_prices)]

            mae = mean_absolute_error(actual_prices, predicted_prices)
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

            comparison_data = pd.DataFrame({
                'Date': prediction_dates.strftime('%Y-%m-%d'),
                'Actual Price': actual_prices,
                'Predicted Price': predicted_prices
            }).to_dict('records')

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=training_data.index, y=training_data['Close'], mode='lines',
                                     name='Historical Price', line=dict(color='#4299e1')))
            fig.add_trace(go.Scatter(x=prediction_dates, y=actual_prices, mode='lines',
                                     name='Actual Price', line=dict(color='#2ecc71', width=3)))
            fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices, mode='lines',
                                     name='Predicted Price', line=dict(color='#e74c3c', width=3)))
            fig.update_layout(title=f"Backtest Results for {ticker} from {backtest_date_str}")

        fig.update_layout(
            xaxis_title=None, yaxis_title='Stock Price', template='plotly_white',
            legend=dict(x=0.01, y=0.99, bordercolor='lightgray', borderwidth=1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        plot_html = fig.to_html(full_html=False, config={'displayModeBar': False})

        company_info = stock.info
        company_name = company_info.get('longName', ticker.upper())
        news_articles = get_news(company_name)

        context = {
            "ticker": ticker,
            "company_name": company_name,
            "plot_html": plot_html,
            "comparison_data": comparison_data,
            "news_articles": news_articles,
            "mode": mode
        }
        if mode == "Backtest":
            context["mae"] = f"{mae:.2f}"
            context["mape"] = f"{mape:.2f}%"

        return render_template('results.html', **context)

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return render_template('results.html', error="An unexpected error occurred. Please try again later.")


if __name__ == '__main__':
    app.run(debug=True)
