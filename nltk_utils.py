# Import Libraries
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import SnowballStemmer
stemmer= SnowballStemmer(language= 'english')
from nltk.corpus import stopwords
nltk.download('stopwords')

# Tokenize text i.e make all text be in a list format e.g "I am sick" = ['i', 'am', 'sick']
def tokenize(text):
  return [stemmer.stem(token) for token in word_tokenize(text)]

# Create stopwords to reduce noise in data
english_stopwords= stopwords.words('english')

# Create a vectosizer to learn all words in order to convert them into numbers
def vectorizer():
    vectorizer= TfidfVectorizer(tokenizer=tokenize,
                                stop_words=english_stopwords,
                                )
    return vectorizer

