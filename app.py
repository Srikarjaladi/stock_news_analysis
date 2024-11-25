from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import yfinance as yf
import io
import base64
import logging
from transformers import BertTokenizer
import torch
from data_loader import StockNewsDataset
from predict import predict
from train import train_model
from dataset import load_and_split_data

# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Load dataset paths
DATASET_FILE = '/Users/srikarjaladi/Desktop/Get_it_done/Fine_tuned_CSV_files/news_with_textblob_sentiments.csv'  # Replace with your dataset path
HEADLINE_DATASET_FILE = '/Users/srikarjaladi/Desktop/Get_it_done/Fine_tuned_CSV_files/all_merged_significant_changes.csv'  # Replace with your dataset path

# Load and preprocess ticker sentiment analysis dataset
df = pd.read_csv(DATASET_FILE)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).sort_values(by='Date')
df.columns = df.columns.str.strip()

# Ensure necessary columns exist
required_columns = ['Ticker', 'Date', 'Title', 'Excerpt', 'Summary']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"The dataset must contain these columns: {required_columns}")

# Load and preprocess headline sentiment prediction dataset
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_split_data(HEADLINE_DATASET_FILE)

# Initialize BERT tokenizer and prepare datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize training dataset
train_texts = [str(text) for text in train_texts]
train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
train_dataset = StockNewsDataset(train_encodings, train_labels.tolist())

# Tokenize validation dataset
val_texts = [str(text) for text in val_texts]
val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
val_dataset = StockNewsDataset(val_encodings, val_labels.tolist())

# Train the BERT model
model = train_model(train_dataset, val_dataset, epochs=4, batch_size=16)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Sentiment analysis function
def analyze_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Generate graph as Base64
def generate_graph_base64(filtered_df, stock_data, ticker):
    if stock_data.empty:
        logging.debug("Stock data is empty.")
        raise ValueError(f"No stock data available for ticker {ticker} in the specified date range.")

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    plt.style.use('ggplot')

    # Stock market graph
    axes[0].plot(stock_data['Date'], stock_data['Close'], color='green', marker='o', linestyle='-', label='Stock Price')
    axes[0].set_title(f'Real-Time Stock Market Graph for {ticker}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Close Price')
    axes[0].legend(loc='upper left')

    # Sentiment analysis graphs
    for i, (col, color, title) in enumerate(zip(
        ['Title Sentiment Score', 'Excerpt Sentiment Score', 'Summary Sentiment Score'],
        ['royalblue', 'orange', 'purple'],
        ['Title Sentiment', 'Excerpt Sentiment', 'Summary Sentiment']
    )):
        axes[i + 1].bar(filtered_df['Date'], filtered_df[col], color=color, alpha=0.8, width=0.4)
        axes[i + 1].set_title(title)
        axes[i + 1].set_ylabel('Sentiment Score')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode to Base64
    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    logging.debug("Graph successfully generated and encoded to Base64.")
    return graph_base64

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    graph_base64 = None
    prediction = None
    error = None

    if request.method == 'POST':
        option = request.form['option']
        if option == 'ticker':
            ticker = request.form['ticker'].strip().upper()
            logging.debug(f"Ticker selected: {ticker}")
            filtered_df = df[df['Ticker'] == ticker]

            if filtered_df.empty:
                error = f"No data found for ticker {ticker}"
                logging.debug(error)
            else:
                # Calculate sentiment scores if not already present
                if 'Title Sentiment Score' not in filtered_df.columns:
                    filtered_df['Title Sentiment Score'] = filtered_df['Title'].apply(analyze_sentiment)
                if 'Excerpt Sentiment Score' not in filtered_df.columns:
                    filtered_df['Excerpt Sentiment Score'] = filtered_df['Excerpt'].apply(analyze_sentiment)
                if 'Summary Sentiment Score' not in filtered_df.columns:
                    filtered_df['Summary Sentiment Score'] = filtered_df['Summary'].apply(analyze_sentiment)

                # Fetch stock data
                start_date = filtered_df['Date'].min().strftime('%Y-%m-%d')
                end_date = filtered_df['Date'].max().strftime('%Y-%m-%d')
                stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')

                if stock_data.empty:
                    error = f"No stock data available for ticker {ticker} between {start_date} and {end_date}."
                    logging.debug(error)
                else:
                    stock_data.reset_index(inplace=True)
                    graph_base64 = generate_graph_base64(filtered_df, stock_data, ticker)

        elif option == 'headline':
            headline = request.form['headline'].strip()
            if headline:
                logging.debug(f"Headline submitted: {headline}")
                prediction = predict(model, tokenizer, headline)
                logging.debug(f"Prediction result: {prediction}")

    return render_template('index.html', graph=graph_base64, prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
