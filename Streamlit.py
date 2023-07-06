#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install streamlit


# In[3]:


import streamlit as st
import pandas as pd

def user_input_features():
        default_value_goes_here="I like the movie"
        review=st.text_input("statement", default_value_goes_here)


        data = {
                'review': review,
                
                }
        features = pd.DataFrame(data, index=[0])
        return features

#App Title
st.title("Sentiment predictor")

#st.write("Hello")

st.write("""
### SENTIMENT PREDICTION
This app analyse the **SENTIMENTS**
""")
st.sidebar.header('User Input Features')
st.subheader('User Input parameters')
df_features = user_input_features()

st.write(df_features)
# In[ ]:




