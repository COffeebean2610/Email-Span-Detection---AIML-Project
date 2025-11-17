import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

print("Loading data...")
# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)

print("Preprocessing data...")
# Encode target
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df = df.drop_duplicates(keep='first')

print(f"Dataset shape after preprocessing: {df.shape}")

# Text preprocessing function
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return ' '.join(y)

print("Transforming text...")
# Apply preprocessing
df['transformed_text'] = df['text'].apply(transform_text)

print("Creating TF-IDF vectors...")
# Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

print(f"Feature matrix shape: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print("Training model...")
# Train model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Evaluate
y_pred = mnb.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')

print("Saving model and vectorizer...")
# Save model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(mnb, f)

print('Model and vectorizer saved successfully!')

# Test loading
print("Testing model loading...")
with open('vectorizer.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Test with a sample
test_text = "Free entry in 2 a wkly comp to win FA Cup final tkts"
transformed_test = transform_text(test_text)
test_vector = loaded_tfidf.transform([transformed_test])
prediction = loaded_model.predict(test_vector)[0]

print(f"Test prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
print("Model loading test successful!")