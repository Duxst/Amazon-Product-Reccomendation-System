import pandas as pd
from textblob import TextBlob
from transformers import pipeline

try:
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('c:\\Users\\Student\\Documents\\GitHub\\AmazonRecommendationSystem\\data\\amazon.csv')
    print("Dataset loaded successfully.")

    # Define the conversion rate
    INR_TO_USD_RATE = 0.012

    # Function to convert price columns from INR to USD
    def convert_inr_to_usd(price):
        # Remove currency symbols and commas, and convert to float
        price = price.replace('₹', '').replace(',', '').strip()
        try:
            return float(price) * INR_TO_USD_RATE
        except ValueError:
            return 0.0

    # Function to perform sentiment analysis on reviews
    def analyze_sentiment(review):
        analysis = TextBlob(review)
        return analysis.sentiment.polarity

    # Initialize summarization pipeline
    print("Loading summarization model...")
    summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")
    print("Summarization model loaded successfully.")

    # Function to summarize reviews
    def summarize_review(review):
        # Truncate reviews longer than the model's maximum input length
        max_input_length = 1024
        if len(review) > max_input_length:
            review = review[:max_input_length]
        # Summarize the review
        try:
            summary = summarizer(review, max_length=50, min_length=25, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            return "Summary not available"

    # Apply the conversion to the price columns
    print("Converting prices to USD...")
    df['discounted_price_usd'] = df['discounted_price'].apply(convert_inr_to_usd)
    df['actual_price_usd'] = df['actual_price'].apply(convert_inr_to_usd)
    print("Price conversion completed.")

    # Apply sentiment analysis to the review content
    print("Performing sentiment analysis...")
    df['review_sentiment'] = df['review_content'].apply(analyze_sentiment)
    print("Sentiment analysis completed.")

    # Apply summarization to the review content
    print("Summarizing reviews...")
    df['review_summary'] = df['review_content'].apply(summarize_review)
    print("Review summarization completed.")

    # Display the updated dataframe
    print(df[['product_name', 'discounted_price_usd', 'actual_price_usd', 'review_sentiment', 'review_summary']].head())

    # Save the updated dataframe to a new CSV file
    df.to_csv('amazon_products_usd_sentiment_summary.csv', index=False)
    print("CSV file created successfully.")

except Exception as e:
    print(f"An error occurred: {e}")
