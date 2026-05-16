import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

with open ('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print(label_encoder.classes_)