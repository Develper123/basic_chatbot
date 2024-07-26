import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load data files with error handling
try:
    with open('intents.json') as file:
        intents = json.load(file)
except FileNotFoundError:
    raise FileNotFoundError("The 'intents.json' file is not found. Please ensure it is in the correct directory.")

try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
except FileNotFoundError:
    raise FileNotFoundError(
        "The 'words.pkl' or 'classes.pkl' file is not found. Please ensure they are in the correct directory.")

try:
    model = load_model('chatbot_model.keras')
except OSError:
    raise OSError(
        "The 'chatbot_model.keras' file is not found or is corrupted. Please ensure it is in the correct directory.")

# Mapping predicted tags to intent.json tags
tag_mapping = {
    'greeting': 'greetings',
    'goodbye': 'goodbye',
    'thank': 'thanks',
    'request_info':'request_info',
    'small_talk': 'small_talk',
    'feedback': 'feedback',
    'report_issue': 'report_issue'

    # Add more mappings if needed
}


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    #print(f"Sentence words after tokenization and lemmatization: {sentence_words}")  # Debug print
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    #print(f"Bag of words: {bag}")  # Debug print
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    #print(f"Bag of words for model prediction: {bow}")  # Debug print
    res = model.predict(np.array([bow]))[0]
    #print(f"Model prediction: {res}")  # Debug print
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    #print(f"Sorted prediction results above threshold: {results}")  # Debug print
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    #print(f"Return list of predicted intents: {return_list}")  # Debug print
    return return_list


def get_response(predicted_intents, intents_json):
    if not predicted_intents:
        return "I'm sorry, I didn't understand that."

    tag = predicted_intents[0]['intent']
    tag = tag_mapping.get(tag, tag)  # Map the predicted tag if needed
    list_of_intents = intents_json['intents']

    #print(f"Predicted tag after mapping: {tag}")  # Debug print
    #print(f"List of intents: {[intent['tag'] for intent in list_of_intents]}")  # Debug print

    result = "I'm sorry, I didn't understand that."  # Default response

    for i in list_of_intents:
        if i['tag'].lower() == tag.lower():  # Case insensitive comparison
            result = random.choice(i['responses'])
            break

    return result


print("GO!Bot is running! (type 'exit' to stop)")

while True:
    message = input("")
    if message.lower() == 'exit':
        print("Goodbye!")
        break

    predicted_intents = predict_class(message)
    response = get_response(predicted_intents, intents)
    print(response)
