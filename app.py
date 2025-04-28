# ========== Imports ========== 
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import openai

# ========== Streamlit Setup (MUST be first) ==========
st.set_page_config(page_title="📈 Stock Screener & Chatbot", layout="wide")

# ========== Custom CSS for Enhanced UI ==========
st.markdown("""
    <style>
    /* General App Styling */
    body {
        background-color: #f0f4f7;
        color: #333;
        font-family: 'Arial', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #0044cc;
        margin-bottom: 40px;
    }
    .subheader {
        color: #0066ff;
    }

    /* Floating Chatbot Styling */
    #floating-chatbot {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 320px;
        background-color: #ffffff;
        border: 2px solid #0044cc;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        z-index: 1000;
        cursor: pointer;
    }
    #floating-chatbot h4 {
        color: #0044cc;
        margin-bottom: 10px;
    }

    /* Form Inputs */
    .stTextInput, .stButton {
        margin-bottom: 20px;
    }

    /* Plots */
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    /* Dataframe */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== OpenAI API Key ==========
openai.api_key = "your-openai-api-key"

# ========== LSTM Model ==========
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 50)
        c0 = torch.zeros(1, x.size(0), 50)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# ========== Functions ==========
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
        
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

def get_stock_data(ticker, start="2022-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)
    return data

def predict_stock(ticker):
    data = get_stock_data(ticker)
    if data.empty:
        return "No data found.", None, None, None

    close_prices = data['Close']
    X, y, scaler = preprocess_data(close_prices)

    model = LSTMModel()
    model.eval()
    last_60_days = torch.tensor(X[-1:]).float()
    prediction = model(last_60_days)
    predicted_price = scaler.inverse_transform(prediction.detach().numpy())

    last_price = close_prices.iloc[-1].item()
    trend = "upward 📈" if predicted_price[0][0] > last_price else "downward 📉"

    return trend, data, close_prices, predicted_price[0][0]

def plot_comparison_charts(main_data, competitors):
    fig, axes = plt.subplots(2, 1, figsize=(12,12))

    # Plot Closing Prices Comparison
    axes[0].plot(main_data['Close'], label=f'Main Stock (Closing)', linewidth=2)
    for comp in competitors:
        data = get_stock_data(comp)
        if not data.empty:
            axes[0].plot(data['Close'], label=f'{comp} (Closing)')

    axes[0].set_title('Stock Price Comparison (Closing Price)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].legend()

    # Plot Opening Prices Comparison
    axes[1].plot(main_data['Open'], label=f'Main Stock (Opening)', linewidth=2)
    for comp in competitors:
        data = get_stock_data(comp)
        if not data.empty:
            axes[1].plot(data['Open'], label=f'{comp} (Opening)')

    axes[1].set_title('Stock Price Comparison (Opening Price)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price')
    axes[1].legend()

    st.pyplot(fig)

def get_quarterly_profit_loss(ticker, competitors):
    data_dict = {}
    try:
        stock = yf.Ticker(ticker)
        quarterly_financials = stock.quarterly_financials.T
        if quarterly_financials.empty:
            return None
        data_dict[ticker] = quarterly_financials[['Total Revenue', 'Gross Profit', 'Net Income']]
        
        for comp in competitors:
            comp_stock = yf.Ticker(comp)
            comp_financials = comp_stock.quarterly_financials.T
            if comp_financials.empty:
                continue
            data_dict[comp] = comp_financials[['Total Revenue', 'Gross Profit', 'Net Income']]

        # Merge all DataFrames into one
        merged_df = pd.concat(data_dict.values(), keys=data_dict.keys(), axis=1)
        
        return merged_df
    except Exception as e:
        return None

# ========== Conclusion Function ==========
def generate_investment_conclusion(ticker, competitors, main_trend, competitor_trends, profit_loss_table):
    conclusion = f"**Investment Conclusion for {ticker.upper()}**\n\n"
    
    conclusion += f"### Trend Analysis\n"
    conclusion += f"- {ticker.upper()} trend: {main_trend} ✅\n" if main_trend == "upward 📈" else f"- {ticker.upper()} trend: {main_trend} ❌\n"
    
    for comp, trend in competitor_trends.items():
        conclusion += f"- {comp} trend: {trend} ✅\n" if trend == "upward 📈" else f"- {comp} trend: {trend} ❌\n"
    
    conclusion += "\n### Quarterly Profit & Loss Analysis\n"
    if profit_loss_table is not None:
        conclusion += f"The quarterly profit and loss data for {ticker.upper()} is as follows:\n"
        st.dataframe(profit_loss_table)

        if 'Net Income' in profit_loss_table.columns:
            best_profit = profit_loss_table.loc[profit_loss_table['Net Income'].idxmax()]
            conclusion += f"Best quarter based on Net Income: {best_profit.name} with a Net Income of ${best_profit['Net Income']:,} ✅\n"
        else:
            conclusion += f"Net Income data is not available in the quarterly profit-loss data. ❌\n"
    else:
        conclusion += f"Profit and loss data for {ticker.upper()} is unavailable or incomplete. ❌\n"
    
    conclusion += "\n### Final Recommendation\n"
    if main_trend == "upward 📈":
        if all([trend == "upward 📈" for trend in competitor_trends.values()]):
            conclusion += f"\nSince both {ticker.upper()} and its competitors show an upward trend, it might be a good time to invest in {ticker.upper()} ✅.\n"
        else:
            conclusion += f"\nAlthough {ticker.upper()} shows an upward trend, some competitors show a downward trend, suggesting some caution in investment. ❌\n"
    else:
        if all([trend == "downward 📉" for trend in competitor_trends.values()]):
            conclusion += f"\nSince both {ticker.upper()} and its competitors show a downward trend, it might be worth reconsidering investment in {ticker.upper()} ❌.\n"
        else:
            conclusion += f"\nDespite {ticker.upper()}'s downward trend, some competitors are showing growth, which may indicate potential opportunities elsewhere. ✅\n"
    
    return conclusion

# ========== Chatbot Section ==========
def get_chatbot_response(user_input):
    """Function to get chatbot response from OpenAI GPT"""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # or another OpenAI model
            prompt=user_input,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Store chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for msg in st.session_state['messages']:
    if msg["role"] == "user":
        st.sidebar.text(f"User: {msg['content']}")
    else:
        st.sidebar.text(f"Bot: {msg['content']}")

# User input for chatbot
user_input = st.sidebar.text_input("Ask a question about the stock", "")

# When the user submits the input
if user_input:
    # Add user message to the chat history
    st.session_state['messages'].append({"role": "user", "content": user_input})

    # Get chatbot's response
    bot_response = get_chatbot_response(user_input)

    # Add bot message to the chat history
    st.session_state['messages'].append({"role": "assistant", "content": bot_response})

    # Display the bot's response in the sidebar
    st.sidebar.text(f"Bot: {bot_response}")

# ========== Streamlit App Layout ==========
st.markdown('<div class="title">📊 Stock Screener & LSTM Trend Predictor</div>', unsafe_allow_html=True)

ticker = st.text_input("Enter main stock ticker (e.g., AAPL, MSFT):")

top_competitors = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'SPY'
]
competitors = st.multiselect("Select competitors (hold CTRL to select multiple)", top_competitors)

if st.button("Predict & Compare"):
    trend, data, close_prices, predicted_price = predict_stock(ticker)

    if data is not None:
        st.success(f"Predicted trend for {ticker.upper()}: {trend}")
        st.metric(label="Predicted next close price", value=f"${predicted_price:.2f}")
        
        plot_comparison_charts(data, competitors)

        profit_loss_table = get_quarterly_profit_loss(ticker, competitors)
        competitor_trends = {comp: predict_stock(comp)[0] for comp in competitors}
        conclusion = generate_investment_conclusion(ticker, competitors, trend, competitor_trends, profit_loss_table)
        st.markdown(conclusion)


