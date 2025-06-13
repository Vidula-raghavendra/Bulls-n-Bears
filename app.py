import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from flask import Flask, request, render_template

# Imports for plotting and news
import plotly.graph_objects as go
from GoogleNews import GoogleNews

# Keras/TensorFlow imports are kept for potential local use with the real model
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
            if title and link and not link.startswith('http'):
                link = f"https://news.google.com{link[1:]}"
            if title and link:
                articles.append({"title": title, "link": link})
        return articles
    except Exception as e:
        app.logger.error(f"Error fetching news for {query}: {e}")
        return []

# ------------------------ Mock Prediction Function (for Deployment) ------------------------
def create_lstm_model_and_predict(data, future_days=7):
    """
    MOCK FUNCTION: Generates a realistic-looking but fake prediction.
    It takes the last day's price and creates small, random daily changes.
    """
    last_price = data['Close'].iloc[-1]
    future_prices = []
    current_price = last_price
    for _ in range(future_days):
        change_percent = np.random.uniform(-0.025, 0.025)
        current_price *= (1 + change_percent)
        future_prices.append(current_price)
    return np.array(future_prices)


# ------------------------ Web App Routes ------------------------
@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and displays prediction results."""
    ticker = request.form.get('ticker')
    if not ticker:
        return render_template('results.html', error="Ticker symbol cannot be empty.")

    try:
        stock = yf.Ticker(ticker)
        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=365 * 2)
        data = stock.history(start=start, end=end)

        if data.empty:
            error_msg = f"No data found for ticker '{ticker}'. This could be an invalid symbol or a delisted stock."
            return render_template('results.html', ticker=ticker, error=error_msg)

        # --- START OF THE ROBUST FIX ---
        # We will try to get the detailed info, but have a fallback if it fails.
        try:
            company_info = stock.info
            # Check if the info dictionary is valid, sometimes it returns just {'regularMarketPrice': None}
            if company_info and company_info.get('logo_url'):
                company_name = company_info.get('longName', ticker.upper())
                sector = company_info.get('sector', 'N/A')
                industry = company_info.get('industry', 'N/A')
                quote_type = company_info.get('quoteType', 'N/A')
            else:
                # If the info is incomplete, raise an exception to go to the fallback plan.
                raise ValueError("Incomplete data from yfinance.info")
        except Exception as e:
            app.logger.warning(f"Could not fetch .info for {ticker}: {e}. Using fallback.")
            # Fallback Plan: Use basic info and defaults.
            company_name = ticker.upper()
            sector = 'N/A'
            industry = 'N/A'
            quote_type = 'N/A' # We don't know the type for sure
        # --- END OF THE ROBUST FIX ---

        news_articles = get_news(company_name if company_name != ticker.upper() else ticker)
        last_10_days = data.tail(10).reset_index()[['Date', 'Close']].to_dict('records')
        last_10_days = [{'index': row['Date'].strftime('%Y-%m-%d'), 'Close': row['Close']} for row in last_10_days]
        
        future_prices = create_lstm_model_and_predict(data)
        future_dates = [(datetime.datetime.today() + datetime.timedelta(days=i + 1)) for i in range(len(future_prices))]

        # Create Plotly graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close', line=dict(color='#4299e1', width=3)))
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Predicted Close', line=dict(color='#f56565', dash='dash', width=3)))
        fig.update_layout(
            title=None, xaxis_title=None, yaxis_title='Stock Price',
            template='plotly_white', legend=dict(x=0.01, y=0.99, bordercolor='lightgray', borderwidth=1),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        plot_html = fig.to_html(full_html=False, config={'displayModeBar': False})

        return render_template('results.html',
                               ticker=ticker,
                               company_name=company_name,
                               sector=sector,
                               industry=industry,
                               quote_type=quote_type,
                               plot_html=plot_html,
                               last_10_days=last_10_days,
                               news_articles=news_articles)

    except Exception as e:
        app.logger.error(f"Major error processing ticker {ticker}: {e}")
        error_message = f"A critical error occurred. Please check the ticker symbol or try again later."
        return render_template('results.html', ticker=ticker, error=error_message)

# ------------------------ Run Server ------------------------
if __name__ == '__main__':
    app.run(debug=True)