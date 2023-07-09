#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install streamlit
pip install ntlk

# In[3]:

import nltk
nltk.download('stopwords')
import streamlit as st
import pandas as pd
import numpy as np
import time 
import re
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
english_stop_words = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from PIL import Image
#preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# machine learning

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


from sklearn.metrics import classification_report
from wordcloud import  WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style ='whitegrid')
pd.set_option('display.max_columns',None)

#MODEL
headers=['Tweet_ID','Entity','Sentiment','Tweet_content']
train_df=pd.read_csv('twitter_training.csv', sep=',', names=headers)

train_df= train_df.drop_duplicates()
train_df.dropna(axis=0, inplace=True)

tweet_train  = train_df["Tweet_content"]
tweet_valid=train_df["Tweet_content"]
target=train_df['Sentiment']

# encoder for target feature
from sklearn import preprocessing
lb = preprocessing.LabelEncoder()
#train_df['Sentiment']=lb.fit_transform(train_df['Sentiment'])
train_df['Sentiment']=train_df['Sentiment'].replace({"Negative":1,"Positive":3,"Neutral":2,"Irrelevant":0})




#Step (1): Remove Additional Letter such as @
REPLACE_WITH_SPACE = re.compile("(@)")
SPACE = " "

def preprocess_reviews(reviews):  
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line.lower()) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(tweet_train)
reviews_valid_clean = preprocess_reviews(tweet_valid)

#Step (2): Remove Stop Words

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()  if word not in english_stop_words]))
    return removed_stop_words
    
no_stop_words_train = remove_stop_words(reviews_train_clean)
no_stop_words_valid = remove_stop_words(reviews_valid_clean)

#Step(3) : Stemming


def get_stemmed_text(corpus):
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews_train = get_stemmed_text(no_stop_words_train)
stemmed_reviews_test = get_stemmed_text(no_stop_words_valid)

#Step(4) : TF-IDF

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(stemmed_reviews_train)
X = tfidf_vectorizer.transform(stemmed_reviews_train)
X_test = tfidf_vectorizer.transform(stemmed_reviews_test)

#Step(5) : Spliting Data
X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

#Step(6) : Machine Learing Model

text_classifier = RandomForestClassifier(n_estimators=500, random_state=0)
text_classifier.fit(X_train, y_train)

y_pred=text_classifier.predict(X_val)
model=text_classifier

#App Title
st.title("SENTIMENT ANALYZER")
image = Image.open('sentiment_analyzer_background.png')
new_image = image.resize((1000, 400))
st.image(new_image)



st.write("""
This app detects the **EMOTIONS** in the data
""")

st.title('Text analyser')
TextBox = st.text_area('Enter text here...')

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)

#model=pickle.load(open('sentimental_analysis_model.pkl', 'rb'))
def prediction(user_input):
    pre_1=preprocess_reviews(list(user_input))
    pre_2=remove_stop_words(pre_1)
    pre_3=get_stemmed_text(pre_2)
    pre_4 = tfidf_vectorizer.transform(pre_3)
    prediction=model.predict(pre_4)
    probability=model.predict_proba(pre_4)[:,1]
    if prediction[0]==1:
        sentiment="Negative"
    elif prediction[0]==3:
        sentiment="Positive"
    elif prediction[0]==2:
        sentiment="Neutral"
    else:
        sentiment="Irrelevant"
    return sentiment,probability[0]
if st.button('Analyse'):
    video_file = open('138115-data-analysis.mp4', 'rb')
    video_bytes = video_file.read()
    
    st.video(video_bytes)
    sentiment,probability=prediction(TextBox)
    st.write(sentiment)
    probability=probability*100
    st.write(probability)

else:
    st.write("Error")






