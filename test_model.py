import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return ' '.join(y)

def main():
    try:
        print("Loading model and vectorizer...")
        
        # Load the vectorizer and model
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print("Model loaded successfully!")
        
        # Test messages
        test_messages = [
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
            "Hey, how are you doing today?",
            "URGENT! You have won a 1000 cash prize. Call now!",
            "Can we meet for lunch tomorrow?",
            "Congratulations! You've won £1000 cash! Call 09061701461 now!"
        ]
        
        print("\nTesting predictions:")
        print("=" * 60)
        
        for i, message in enumerate(test_messages, 1):
            # Transform the message
            transformed = transform_text(message)
            
            # Vectorize
            vector = tfidf.transform([transformed])
            
            # Predict
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]
            
            print(f"Test {i}:")
            print(f"Message: {message}")
            print(f"Transformed: {transformed}")
            print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
            print(f"Confidence: {max(probability):.3f}")
            print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("All tests passed!")
    else:
        print("Tests failed!")