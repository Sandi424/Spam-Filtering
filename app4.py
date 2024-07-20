import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Streamlit settings
st.set_page_config(page_title="Spam Filtering", layout="centered")
st.title("Spam Filtering Project")
st.markdown("""
    Enter a message to check if it's SPAM or NOT SPAM.
    """)

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


stopwords = ['u', '2', 'ur', "i'm", '4', '...',
             'ok', "i'll"] + list(string.punctuation)

data['message'] = data['message'].apply(lambda x: process_string(x, stopwords))

# Model training
X = data['message']
Y = data['label']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting new messages
st.subheader("Predict New Message")
new_message = st.text_input("Enter a message to check if it's spam:")
if st.button("Predict"):
    if new_message:
        new_message_transformed = vectorizer.transform([new_message])
        prediction = model.predict(new_message_transformed)
        if prediction[0] == 'spam':
            st.error("SPAM")
        else:
            st.success("NOT SPAM")
    else:
        st.warning("Please enter a message.")
