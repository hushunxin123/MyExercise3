import nltk
from textblob import TextBlob

# Load the Moby Dick text
moby_dick = nltk.corpus.gutenberg.words('melville-moby_dick.txt')
text = ' '.join(moby_dick)

# Performing sentiment analysis
blob = TextBlob(text)
sentiment_score = blob.sentiment.polarity
rounded_score = round(sentiment_score, 3)  # Rounding the sentiment score to three decimal places

# Displaying the average sentiment score
print("Average Sentiment Score:", rounded_score)

# Determining the overall text sentiment
if sentiment_score > 0.05:
    print("Overall Text Sentiment: positive")
else:
    print("Overall Text Sentiment: negative")


