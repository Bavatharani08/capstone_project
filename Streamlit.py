#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install streamlit


# In[3]:


import streamlit as st
import pandas as pd
from PIL import Image


#App Title
st.title("SENTIMENT ANALYZER")
image = Image.open('sentiment_analyzer_background.png')
new_image = image.resize((1000, 400))
st.image(new_image)

#st.write("Hello")

st.write("""
This app detects the **EMOTIONS** in the data
""")
#st.sidebar.header('User Input Features')
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

b = st.button("Analyse")
# if st.button('Analyse'):
#     st.write(TextBox)


# In[ ]:




