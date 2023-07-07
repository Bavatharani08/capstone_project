#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install streamlit


# In[3]:


import streamlit as st
import pandas as pd
from PIL import Image


#App Title
st.title("Sentiment Analyzer")
image = Image.open('sentiment_analyzer_background.png')
new_image = image.resize((1000, 400))
st.image(new_image)

#st.write("Hello")

st.write("""
### SENTIMENT ANALYZER
This app detects the **EMOTIONS** in the data
""")
#st.sidebar.header('User Input Features')
st.title('Text analyser')
TextBox = st.text_area('Enter text here...')

if st.button('Analyse'):
    st.write(TextBox)


# In[ ]:




