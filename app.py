import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load the model and tokenizer from the unzipped directories
model = AutoModelForSequenceClassification.from_pretrained('./senti_model')
tokenizer = AutoTokenizer.from_pretrained('./senti_model_tokenization')

# Define the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define the predict function
def predict(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    return preds.cpu().numpy()

# Streamlit UI
st.title("Disaster Tweet Prediction")
st.write("Enter a tweet to predict if it indicates a disaster or not.")

# Input text
user_input = st.text_area("Tweet Text")

# Button to make prediction
if st.button("Predict"):
    if user_input:
        prediction = predict([user_input])[0]
        label = "Disaster" if prediction == 1 else "Non-Disaster"
        st.write(f"The tweet indicates a: {label}")
    else:
        st.write("Please enter a tweet.")
