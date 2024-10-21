import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Streamlit settings
st.set_page_config(page_title="Spam Filtering", layout="centered")
st.markdown("""
    <style>
    body {
        background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20230611/pngtree-spam-security-warning-on-red-binary-technology-background-spam-security-warning-photo-image_3043213.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .stApp {
        background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent black overlay */
    }
    .stTextInput>div>div>input {
        background-color: white;
        color: black;
        border: 2px solid #00FF00;
        border-radius: 5px;
        padding: 10px;
    }
    .stTextInput>div>div>input::placeholder {
        color: #A9A9A9;
        font-style: italic;
    }
    .stButton>button {
        background-color: #0000FF;
        color: white;
        border: 2px solid white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px #0000FF;
        display: block;
        margin: 20px auto; /* Center the button */
    }
    .stButton>button:hover {
        background-color: #1E90FF;
        box-shadow: 0 0 20px #1E90FF;
    }
    .stMarkdown {
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: white;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 20px;
    }
    .spam-box, .not-spam-box {
        margin: 20px auto; /* Center the message box */
    }
    .spam-box {
        background-color: #FF0000;
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 15px #FF0000;
        width: 200px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }
    .not-spam-box {
        background-color: #00FF00;
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 15px #00FF00;
        width: 200px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title centered and extra bold
st.markdown("<div class='title'>SPAM PREDICTOR</div>", unsafe_allow_html=True)

# Load data
data = pd.read_csv('spam.csv', usecols=[0, 1], encoding='latin')
data.rename({'v1': 'label', 'v2': 'message'}, axis=1, inplace=True)

# Preprocess data
def process_string(msg, stopwords=[]):
    msg = msg.lower()
    sentence = [word for word in msg.split() if word not in stopwords]
    msg = " ".join(sentence)
    msg = re.sub(r"[!\"#$%&\'()*+,-.:;<=>?@[\\\]^_`{|}~]", "", msg)
    return msg

stopwords = ['u', '2', 'ur', "i'm", '4', '...', 'ok', "i'll"] + list(string.punctuation)
data['message'] = data['message'].apply(lambda x: process_string(x, stopwords))

# Model training
X = data['message']
Y = data['label']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting new messages
new_message = st.text_input("", placeholder="Enter a message")
predict_button = st.button("Predict", key="predict_button")

if predict_button:
    if new_message:
        new_message_transformed = vectorizer.transform([new_message])
        prediction = model.predict(new_message_transformed)
        if prediction[0] == 'spam':
            st.markdown("<div class='spam-box'>SPAM</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='not-spam-box'>NOT SPAM</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a message.")
