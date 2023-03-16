import pickle
import random
import json
import numpy as np
import string
import nltk

from keras.models import load_model

intents = json.loads(open('./intents_vn.json', encoding='utf-8').read())

words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))
model = load_model('./chatbot_model.h5')


# Remove punctuation
def remove_punctuation(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence


def clean_up_sentence(sentence):
    sentence = remove_punctuation(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for i, word in enumerate(words):
        if word in sentence_words:
            bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    try:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
                break
    except IndexError:
        result = "I don't understand"
    return result


# stop = False
# while not stop:
#     message = input("Enter message: ")
#     if message in ['end', 'stop']:
#         stop = True
#     else:
#         pred_intents = predict_class(message)
#         res = get_response(pred_intents, intents)
#         print(res)
