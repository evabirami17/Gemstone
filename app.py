import numpy as np
import pickle
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

model1 = keras.models.load_model('model/model.h5')


def reserve(feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10):
  prediction=model1.predict([[feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10]])
  print(prediction)
  return prediction

def main():
   st.title("GemStone Prediction")

   feature1 = st.number_input("ID", value=0.0)
   feature2 = st.number_input("CARAT", value=0.0)
   feature3 = st.number_input("CUT", value=0.0)
   feature4 = st.number_input("COLOR", value=0.0)
   feature5 = st.number_input("CLARITY", value=0.0)
   feature6 = st.number_input("DEPTH", value=0.0)
   feature7 = st.number_input("TABLE", value=0.0)
   feature8 = st.number_input("X", value=0.0)
   feature9 = st.number_input("Y", value=0.0)
   feature10 = st.number_input("Z", value=0.0)
 
   result=''
   if st.button("Predict"):
    result=reserve(feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10)
   st.success('The output is {}'.format(result))
if __name__=='__main__':
  main()
