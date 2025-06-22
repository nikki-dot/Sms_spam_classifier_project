import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

#function of preprocessing
def transform_msg(Message):
    Message=Message.lower()
    Message=nltk.word_tokenize(Message)
    y=[]
    for i in Message:
        if i.isalnum():
            y.append(i)
    Message = y[:]
    y.clear()
    for i in Message:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    Message = y[:]
    y.clear()
    for i in Message:
        y.append(ps.stem(i))
    return " ".join(y)

tf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('SMS Spam Classifier')
input_sms = st.text_input('Enter the message')

if st.button('predict'):
    # 1. preprocess
    transformed_msg = transform_msg(input_sms)
    # 2. vectorize
    vector_input = tf.transform([transformed_msg])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
