import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd
import numpy as np
import re  
from sklearn.preprocessing import LabelEncoder 

st.title("Language Detection")

@st.cache
def load_data():
    path=r"/app/Language-Detection/LanguageDetection.csv"
    data = pd.read_csv(path)
    X = data["Text"]
    y = data["Language"]
    pickl=pd.read_pickle(r"/app/Language-Detection/lang_detection.pkl")

    return (X,y,pickl)

X,y,pkl=load_data()
le=LabelEncoder()

y = le.fit_transform(y)
data_list = []
# iterating through all the text
for text in X:         
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)      # removing the symbols and numbers
    text = re.sub(r'[[]]', ' ', text)   
    text = text.lower()          # converting the text to lower case
    data_list.append(text)       # appending to data_list
cv=CountVectorizer()
X = cv.fit_transform(data_list).toarray()


def predicts(text):
    x=cv.transform([text]).toarray()    
    lang = pkl.predict(x)
    lang = le.inverse_transform(lang)
    return lang[0]

t=st.text_input("Enter the a sentence ")
submit=st.button("hit me")
if submit:
    st.write(predicts(t))    
