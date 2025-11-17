import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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

    return " ".join(y)

# Load model and vectorizer with error handling
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title("📧 Email/SMS Spam Classifier")
st.markdown("---")

# About Spam section
st.header("🚨 What is Spam?")
st.write("""
Spam messages are unsolicited communications sent in bulk, typically for commercial purposes. 
Common characteristics include:
- **Prize/lottery scams**: "You've won £1000!"
- **Urgent calls to action**: "Call now!", "Limited time!"
- **Suspicious phone numbers**: Premium rate numbers
- **Poor grammar and spelling**
- **Requests for personal information**
""")

# Project Info section
st.header("🔬 How This Project Works")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset")
    st.write("• 5,169 SMS messages")
    st.write("• 87% Ham (legitimate)")
    st.write("• 13% Spam messages")
    
    st.subheader("🛠️ Technologies Used")
    st.write("• **Python** - Programming language")
    st.write("• **Streamlit** - Web interface")
    st.write("• **NLTK** - Text processing")
    st.write("• **Scikit-learn** - Machine learning")

with col2:
    st.subheader("🧠 Model Details")
    st.write("• **Algorithm**: Multinomial Naive Bayes")
    st.write("• **Accuracy**: 97.1%")
    st.write("• **Precision**: 100%")
    st.write("• **Features**: TF-IDF (3000 features)")
    
    st.subheader("⚙️ Text Processing")
    st.write("• Lowercasing")
    st.write("• Tokenization")
    st.write("• Remove stopwords")
    st.write("• Stemming")

st.markdown("---")

# Prediction section
st.header("🔍 Test the Classifier")
st.write("Enter any message below to check if it's spam or legitimate:")

input_sms = st.text_area("Enter your message here:", height=100, placeholder="Type your message...")

if st.button('🔍 Classify Message', type="primary"):
    if input_sms.strip():
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            confidence = model.predict_proba(vector_input)[0].max()
            
            # 4. Display results
            st.markdown("### 📊 Results:")
            if result == 1:
                st.error("🚨 **SPAM DETECTED**")
                st.write(f"**Confidence**: {confidence:.1%}")
                st.warning("⚠️ This message shows characteristics of spam. Be cautious!")
            else:
                st.success("✅ **LEGITIMATE MESSAGE**")
                st.write(f"**Confidence**: {confidence:.1%}")
                st.info("✨ This message appears to be safe.")
                
            # Show processed text
            with st.expander("🔧 See processed text"):
                st.code(transformed_sms)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.warning("⚠️ Please enter a message to classify.")

# Examples section
st.markdown("---")
st.header("💡 Try These Examples")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚨 Spam Examples")
    if st.button("Free prize winner!"):
        st.text_area("Message:", "Congratulations! You've won £1000 cash! Call now!", key="spam1")
    if st.button("Urgent offer"):
        st.text_area("Message:", "URGENT! Limited time offer. Text WIN to 12345", key="spam2")

with col2:
    st.subheader("✅ Legitimate Examples")
    if st.button("Casual message"):
        st.text_area("Message:", "Hey, how are you doing today?", key="ham1")
    if st.button("Meeting request"):
        st.text_area("Message:", "Can we meet for lunch tomorrow?", key="ham2")
