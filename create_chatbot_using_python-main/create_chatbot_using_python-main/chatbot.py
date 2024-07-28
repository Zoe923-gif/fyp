import random
import json
import pickle
import numpy as np
import re
import streamlit as st
from keras.models import load_model
from PIL import Image
from nltk.stem import WordNetLemmatizer
from pymongo import MongoClient
from bson import ObjectId

# Ensure NLTK data files are available
import nltk
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents and model
try:
    with open('intents.json') as file:
        intents = json.load(file)
    
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
except Exception as e:
    st.error(f"Error loading resources: {e}")

import tensorflow as tf

# Load the old H5 model
model = tf.keras.models.load_model('C:/Users/zoezh/Downloads/mscnn_fabric_defect_detector.h5')

# Save it in the new Keras format
model.save('C:/Users/zoezh/Downloads/mscnn_fabric_defect_detector.keras')


# Load fabric defect detection model
try:
    fabric_model = tf.keras.models.load_model('C:/Users/zoezh/Downloads/mscnn_fabric_defect_detector.keras')

except Exception as e:
    st.error(f"Error loading fabric defect detection model: {e}")


# MongoDB connection
try:
    client = MongoClient('mongodb+srv://g43:jav3166@cluster0.3s5djtl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')  # Update with your MongoDB URI
    db = client['chatbot']  # Update with your database name
    orders_collection = db['order']  # Update with your orders collection name
    items_collection = db['clothes']  # Update with your items collection name
except Exception as e:
    st.error(f"Error connecting to MongoDB: {e}")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    try:
        res = model.predict(np.array([bow]))[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return []

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

def check_order_number(message):
    order_number_pattern = r'T\d{6}'
    match = re.search(order_number_pattern, message)
    if match:
        order_id = match.group()
        order = orders_collection.find_one({'order_id': order_id})
        if order:
            customer_name = order.get('customer_name', 'N/A')
            item_purchased_ids = order.get('item_purchased', [])
            total_price = order.get('total_price', 'N/A')
            date = order.get('date', 'N/A')
            
            item_details_list = []
            for item_id in item_purchased_ids:
                try:
                    item = items_collection.find_one({'_id': ObjectId(item_id)})
                    if item:
                        item_details = f"\n\nCollection: {item.get('collection', 'N/A')}\nName: {item.get('name', 'N/A')}\nSize: {item.get('size', 'N/A')}\nUnit Price: {item.get('unitPrice', 'N/A')}\nColor: {item.get('color', 'N/A')}"
                        item_details_list.append(item_details)
                    else:
                        item_details_list.append("\nItem not found in the database.")
                except Exception as e:
                    item_details_list.append(f"\nError retrieving item: {e}")

            item_details_str = "".join(item_details_list)
            return f"Order number {order_id} found.\nCustomer Name: {customer_name}\nTotal Price: {total_price}\nDate: {date}{item_details_str}"
        else:
            return f"Order number {order_id} not found in our database."
    return None

def preprocess_image(img):
    img = img.convert("RGB")  
    img = img.resize((128, 128)) 
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def predict_defect(img):
    img_array = preprocess_image(img)
    try:
        prediction = fabric_model.predict(img_array)
        return prediction
    except MemoryError as e:
        st.error(f"MemoryError during fabric defect prediction: {e}")
        return None
    except Exception as e:
        st.error(f"Error during fabric defect prediction: {e}")
        return None

# Streamlit App
st.title("Order Query Chatbot and Fabric Defect Detector")

# Image upload
uploaded_image = st.file_uploader("Upload a fabric image for defect detection", type=['png', 'jpg', 'jpeg'])
if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        if st.button("Detect Defect"):
            prediction = predict_defect(image)
            if prediction is not None:
                if prediction[0][0] > 0.5:  # Example threshold for binary classification
                    st.text("The image shows a defect.")
                else:
                    st.text("The image appears to be defect-free.")
            else:
                st.text("Error in defect detection.")
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Chatbot interface
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        order_response = check_order_number(user_input)
        if order_response:
            st.text_area("Chatbot:", order_response, height=200)
        else:
            ints = predict_class(user_input)
            res = get_response(ints, intents)
            st.text_area("Chatbot:", res, height=200)
    else:
        st.warning("Please enter a message.")