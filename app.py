import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#  Use uploaded nltk_data folder
nltk.data.path.append('./nltk_data')

# Initialize stemmer
ps = PorterStemmer()

# Load stopwords once to speed up preprocessing
stop_words = set(stopwords.words('english'))

# Preprocessing function
def transform_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    text = nltk.word_tokenize(text)
    
    # 3. Remove non-alphanumeric characters
    tokens = [i for i in text if i.isalnum()]
    
    # 4. Remove stopwords and punctuation
    tokens = [i for i in tokens if i not in stop_words and i not in string.punctuation]
    
    # 5. Stemming
    tokens = [ps.stem(i) for i in tokens]
    
    return " ".join(tokens)

#  Cache model and vectorizer for faster reload
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

# Optional spinner for cold start
with st.spinner('Loading model and resources...'):
    tfidf, model = load_model()

# Streamlit UI
st.title('SMS Spam Classifier')

input_sms = st.text_area('Enter your message')

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess
        transform_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transform_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result with emoji
        if result == 1:
            st.header('🚨 Message is spam')
        else:
            st.header('Message is not spam')
