# Import and class names setup
import gradio as gr
import os
import torch
import random
import nltk_utils
import pandas as pd
from sklearn.model_selection import train_test_split
import time

from model import RNN_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Import data
df= pd.read_csv('Symptom2Disease.csv')
df.drop('Unnamed: 0', axis= 1, inplace= True)

# Preprocess data
df.drop_duplicates(inplace= True)
train_data, test_data= train_test_split(df, test_size=0.15, random_state=42 )

# Setup class names
class_names= {0: 'Acne',
              1: 'Arthritis',
              2: 'Bronchial Asthma',
              3: 'Cervical spondylosis',
              4: 'Chicken pox',
              5: 'Common Cold',
              6: 'Dengue',
              7: 'Dimorphic Hemorrhoids',
              8: 'Fungal infection',
              9: 'Hypertension',
              10: 'Impetigo',
              11: 'Jaundice',
              12: 'Malaria',
              13: 'Migraine',
              14: 'Pneumonia',
              15: 'Psoriasis',
              16: 'Typhoid',
              17: 'Varicose Veins',
              18: 'allergy',
              19: 'diabetes',
              20: 'drug reaction',
              21: 'gastroesophageal reflux disease',
              22: 'peptic ulcer disease',
              23: 'urinary tract infection'
              }

vectorizer= nltk_utils.vectorizer()
vectorizer.fit(train_data.text)



# Model and transforms preparation
model= RNN_model()
# Load state dict
model.load_state_dict(torch.load(
    f= '/content/Models/pretrained_symtom_to_disease_model.pth',
    map_location= torch.device('cpu')
    )
)
# Disease Advice
disease_advice = {
    'Acne': "Maintain a proper skincare routine, avoid excessive touching of the affected areas, and consider using over-the-counter topical treatments. If severe, consult a dermatologist.",
    'Arthritis': "Stay active with gentle exercises, manage weight, and consider pain-relief strategies like hot/cold therapy. Consult a rheumatologist for tailored guidance.",
    'Bronchial Asthma': "Follow prescribed inhaler and medication regimen, avoid triggers like smoke and allergens, and have an asthma action plan. Regular check-ups with a pulmonologist are important.",
    'Cervical spondylosis': "Maintain good posture, do neck exercises, and use ergonomic support. Physical therapy and pain management techniques might be helpful.",
    'Chicken pox': "Rest, maintain hygiene, and avoid scratching. Consult a doctor for appropriate antiviral treatment.",
    'Common Cold': "Get plenty of rest, stay hydrated, and consider over-the-counter remedies for symptom relief. Seek medical attention if symptoms worsen or last long.",
    'Dengue': "Stay hydrated, rest, and manage fever with acetaminophen. Seek medical care promptly, as dengue can escalate quickly.",
    'Dimorphic Hemorrhoids': "Follow a high-fiber diet, maintain good hygiene, and consider stool softeners. Consult a doctor if symptoms persist.",
    'Fungal infection': "Keep the affected area clean and dry, use antifungal creams, and avoid sharing personal items. Consult a dermatologist if it persists.",
    'Hypertension': "Follow a balanced diet, exercise regularly, reduce salt intake, and take prescribed medications. Regular check-ups with a healthcare provider are important.",
    'Impetigo': "Keep the affected area clean, use prescribed antibiotics, and avoid close contact. Consult a doctor for proper treatment.",
    'Jaundice': "Get plenty of rest, maintain hydration, and follow a doctor's advice for diet and medications. Regular monitoring is important.",
    'Malaria': "Take prescribed antimalarial medications, rest, and manage fever. Seek medical attention for severe cases.",
    'Migraine': "Identify triggers, manage stress, and consider pain-relief medications. Consult a neurologist for personalized management.",
    'Pneumonia': "Follow prescribed antibiotics, rest, stay hydrated, and monitor symptoms. Seek immediate medical attention for severe cases.",
    'Psoriasis': "Moisturize, use prescribed creams, and avoid triggers. Consult a dermatologist for effective management.",
    'Typhoid': "Take prescribed antibiotics, rest, and stay hydrated. Dietary precautions are important. Consult a doctor for proper treatment.",
    'Varicose Veins': "Elevate legs, exercise regularly, and wear compression stockings. Consult a vascular specialist for evaluation and treatment options.",
    'allergy': "Identify triggers, manage exposure, and consider antihistamines. Consult an allergist for comprehensive management.",
    'diabetes': "Follow a balanced diet, exercise, monitor blood sugar levels, and take prescribed medications. Regular visits to an endocrinologist are essential.",
    'drug reaction': "Discontinue the suspected medication, seek medical attention if symptoms are severe, and inform healthcare providers about the reaction.",
    'gastroesophageal reflux disease': "Follow dietary changes, avoid large meals, and consider medications. Consult a doctor for personalized management.",
    'peptic ulcer disease': "Avoid spicy and acidic foods, take prescribed medications, and manage stress. Consult a gastroenterologist for guidance.",
    'urinary tract infection': "Stay hydrated, take prescribed antibiotics, and maintain good hygiene. Consult a doctor for appropriate treatment."
}



# create gradio app
title= 'Symptom To Disease Prediction App '
description= 'A Natural Language App That can predict diseases based on your symptoms'
article= 'Created at [To be uploaded].'

# Create the gradio demo
with gr.Blocks() as demo:
  chatbot = gr.Chatbot()
  msg = gr.Textbox()
  clear = gr.ClearButton([msg, chatbot])

  def respond(message, chat_history):
    greetings = [
        "hello!",'hello', 'hii !', 'hi', "hi there!",  "hi there!", "heyy", 'good morning', 'good afternoon', 'good evening'
        "hey", "how are you", "how are you?", "how is it going", "how is it going?",
        "what's up?", "how are you?",
        "hey, how are you?", "what is popping"
        "good to see you!", "howdy!",
        "hi, nice to meet you.", "hiya!",
        "hi", "hi, what's new?",
        "hey, how's your day?", "hi, how have you been?", "greetings",
        ]
    responses = [
        "Thank you for using our medical chatbot. Please provide the symptoms you're experiencing, and I'll do my best to predict the possible disease.",
        "Hello! I'm here to help you with medical predictions based on your symptoms. Please describe your symptoms in as much detail as possible.",
        "Greetings! I am a specialized medical chatbot trained to predict potential diseases based on the symptoms you provide. Kindly list your symptoms explicitly.",
        "Welcome to the medical chatbot. To assist you accurately, please share your symptoms in explicit detail.",
        "Hi there! I'm a medical chatbot specialized in analyzing symptoms to suggest possible diseases. Please provide your symptoms explicitly.",
        "Hey! I'm your medical chatbot. Describe your symptoms with as much detail as you can, and I'll generate potential disease predictions.",
        "How can I assist you today? I'm a medical chatbot trained to predict diseases based on symptoms. Please be explicit while describing your symptoms.",
        "Hello! I'm a medical chatbot capable of predicting diseases based on the symptoms you provide. Your explicit symptom description will help me assist you better.",
        "Greetings! I'm here to help with medical predictions. Describe your symptoms explicitly, and I'll offer insights into potential diseases.",
        "Hi, I'm the medical chatbot. I've been trained to predict diseases from symptoms. The more explicit you are about your symptoms, the better I can assist you.",
        "Hi, I specialize in medical predictions based on symptoms. Kindly provide detailed symptoms for accurate disease predictions.",
        "Hello! I'm a medical chatbot with expertise in predicting diseases from symptoms. Please describe your symptoms explicitly to receive accurate insights.",
        ]
    goodbyes = [
        "farewell!",'bye', 'goodbye','good-bye', 'good bye', 'bye', 'thank you', 'later', "take care!",
        "see you later!", 'see you', 'see ya', 'see-you', 'thanks', 'thank', 'bye bye', 'byebye'
        "catch you on the flip side!", "adios!",
        "goodbye for now!", "till we meet again!",
        "so long!", "hasta la vista!",
        "bye-bye!", "keep in touch!",
        "toodles!", "ciao!",
        "later, gator!", "stay safe and goodbye!",
        "peace out!", "until next time!", "off I go!",
        ]
    goodbye_replies = [
        "Take care of yourself! If you have more questions, don't hesitate to reach out.",
        "Stay well! Remember, I'm here if you need further medical advice.",
        "Goodbye for now! Don't hesitate to return if you need more information in the future.",
        "Wishing you good health ahead! Feel free to come back if you have more concerns.",
        "Farewell! If you have more symptoms or questions, don't hesitate to consult again.",
        "Take care and stay informed about your health. Feel free to chat anytime.",
        "Bye for now! Remember, your well-being is a priority. Don't hesitate to ask if needed.",
        "Have a great day ahead! If you need medical guidance later on, I'll be here.",
        "Stay well and take it easy! Reach out if you need more medical insights.",
        "Until next time! Prioritize your health and reach out if you need assistance.",
        "Goodbye! Your health matters. Feel free to return if you have more health-related queries.",
        "Stay healthy and stay curious about your health! If you need more info, just ask.",
        "Wishing you wellness on your journey! If you have more questions, I'm here to help.",
        "Take care and remember, your health is important. Don't hesitate to reach out if needed.",
        "Goodbye for now! Stay informed and feel free to consult if you require medical advice.",
        "Stay well and stay proactive about your health! If you have more queries, feel free to ask.",
        "Farewell! Remember, I'm here whenever you need reliable medical information.",
        "Bye for now! Stay vigilant about your health and don't hesitate to return if necessary.",
        "Take care and keep your well-being a priority! Reach out if you have more health questions.",
        "Wishing you good health ahead! Don't hesitate to chat if you need medical insights.",
        "Goodbye! Stay well and remember, I'm here to assist you with medical queries.",
    ]

    
    if message.lower() in greetings:
      bot_message= random.choice(responses)
    elif message.lower() in goodbyes:
      bot_message= random.choice(goodbye_replies)
    else:
      transform_text= vectorizer.transform([message])
      transform_text= torch.tensor(transform_text.toarray()).to(torch.float32)
      model.eval()
      with torch.inference_mode():
        y_logits=model(transform_text)
        pred_prob= torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
   
      test_pred= class_names[pred_prob.item()] 
      bot_message = f' Based on your symptoms, I believe you are having {test_pred} and I would advice you {disease_advice[test_pred]}'
    chat_history.append((message, bot_message))
    time.sleep(2)
    return "", chat_history

  msg.submit(respond, [msg, chatbot], [msg, chatbot])
# Launch the demo
demo.launch()
