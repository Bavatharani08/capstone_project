#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install streamlit


# In[3]:


import streamlit as st
import pandas as pd
from PIL import Image

def user_input_features():
        default_value_goes_here="Enter the statement"
        review=st.text_input("statement", default_value_goes_here)


        data = {
                'review': review,
                
                }
        features = pd.DataFrame(data, index=[0])
        return review

#App Title
st.title("Sentiment Analyzer")
image = Image.open('sentiment_analyzer_background.png')
new_image = image.resize((1000, 400))
st.image(new_image)

#st.write("Hello")

st.write("""
### SENTIMENT ANALYZER
This app analyses the **SENTIMENTS**
""")
st.sidebar.header('User Input Features')
st.subheader('User Input parameters')
df_features = user_input_features()

st.write(df_features)
# In[ ]:




