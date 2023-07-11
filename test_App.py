#!/usr/bin/env python
# coding: utf-8

# In[25]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

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


# In[26]:


model=pickle.load(open(r'C:\Users\BavatharaniV\Downloads\sentiment_model.sav', 'rb'))
headers=['Tweet_ID','Entity','Sentiment','Tweet_content']
train_df=pd.read_csv(r'C:\Users\BavatharaniV\Downloads\twitter_training.csv', sep=',', names=headers)

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



# In[22]:


#Step (1): Remove Additional Letter such as @
REPLACE_WITH_SPACE = re.compile("(@)")
SPACE = " "

def preprocess_reviews(reviews):  
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line.lower()) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(tweet_train)

#Step (2): Remove Stop Words

def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()  if word not in english_stop_words]))
    return removed_stop_words

no_stop_words_train = remove_stop_words(reviews_train_clean)

#Step(3) : Stemming


def get_stemmed_text(corpus):
    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews_train = get_stemmed_text(no_stop_words_train)


#Step(4) : TF-IDF

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(stemmed_reviews_train)


# In[23]:


def prediction(user_input):
    input_str=user_input.split(" ")
    pre_1=preprocess_reviews(input_str)
    pre_2=remove_stop_words(pre_1)
    pre_3=get_stemmed_text(pre_2)
    pre_4=tfidf_vectorizer.transform(pre_3)
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


# In[24]:


prediction("i like movie")


# In[ ]:


def main():
    #App Title
    st.title("SENTIMENT ANALYZER")
    image = Image.open(r'C:\Users\BavatharaniV\Downloads\sentiment_analyzer_background.png')
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
    
    if st.button('Analyse'):
        #video_file = open(r'C:\Users\BavatharaniV\Downloads\138115-data-analysis.mp4', 'rb')
        #video_bytes = video_file.read()

        #st.video(video_bytes)
        sentiment,probability=prediction(TextBox)
        st.write(sentiment)
        probability=probability*100
        st.write(probability)

    else:
        st.write("Error")


# In[ ]:


if __name__=='__main__':
    main()


# In[ ]:




