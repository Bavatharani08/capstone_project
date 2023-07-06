#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install streamlit


# In[3]:


import streamlit as st

def user_input_features():
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
### Mobile Price Prediction App
This app predicts the **Mobile Price Range**
""")
st.sidebar.header('User Input Features')
st.subheader('User Input parameters')
df_features = user_input_features()

st.write(df_features)
# In[ ]:




