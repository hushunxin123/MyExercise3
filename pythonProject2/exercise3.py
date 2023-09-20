import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Read Moby Dick file from the Gutenberg dataset
moby_dick = nltk.corpus.gutenberg.words('melville-moby_dick.txt')


# Tokenization
def tokenize(text):
    return [word.lower() for word in text if word.isalpha()]  # Words are converted to lower case


tokens = tokenize(moby_dick)


# Stop-words filtering
def filter_stop_words(tokens, language='english'):
    stop_words = set(stopwords.words(language))
    return [word for word in tokens if word not in stop_words]


filtered_tokens = filter_stop_words(tokens)


# Parts-of-Speech tagging
def pos_tag(tokens):
    pos_tags = nltk.pos_tag(tokens)
    return pos_tags


pos_tags = pos_tag(filtered_tokens)


# Convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(nltk_pos):
    if nltk_pos.startswith('J'):
        return 'a'  # adjective
    elif nltk_pos.startswith('V'):
        return 'v'  # verb
    elif nltk_pos.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # noun


# POS frequency
pos_counts = FreqDist([tag for (word, tag) in pos_tags])
top_pos = pos_counts.most_common(5)
print("Top 5 most common parts of speech:")
for pos in top_pos:
    print(pos[0], pos[1])

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for (word, pos) in pos_tags[:20]]
print("\nLemmatized tokens:")
print(lemmatized_tokens)

# Bar chart for pos and their frequencies
plt.bar(pos_counts.keys(), pos_counts.values())
plt.xlabel('Parts-of-Speech')
plt.ylabel('Total Counts')
plt.show()
