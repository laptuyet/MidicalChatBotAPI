
import json
import random
import string
import nltk
import numpy as np
import pickle
import tensorflow as tf

nltk.download('punkt')


intents = json.loads(open('./intents_vn.json', encoding='utf-8').read())

words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))
model = tf.keras.models.load_model('./chatbot_model.h5')


# Remove punctuation
def remove_punctuation(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence


def get_stopwords_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [sw.strip() for sw in stopwords]
        return stopwords


# Load stop words
stop_word_file_path = './vietnamese_stop_words.txt'
stop_words = get_stopwords_list(stop_word_file_path)


def remove_stop_word_from_pattern(sentence):
    for sw in stop_words:
        if ' ' + sw + ' ' in sentence:
            sentence = sentence.replace(sw, '')
    return sentence


def clean_up_sentence(sentence):
    sentence = remove_punctuation(sentence)
    sentence = remove_stop_word_from_pattern(sentence)
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

def getHeartRateType(heart_rate):
    if heart_rate > 100:
        return 'high_bpm'
    elif 65 <= heart_rate <= 100:
        return 'normal_bpm'
    else:
        return 'low_bpm'

def getSpO2Type(spo2):
    if 95 <= spo2 <= 100:
        return 'normal_spo2'
    elif 90 <= spo2 < 95:
        return 'warn_spo2'
    else:
        return 'low_spo2'

# stop = False
# while not stop:
#     message = input("Enter message: ")
#     if message in ['end', 'stop']:
#         stop = True
#     else:
#         flag = message.__contains__('BPM') and message.__contains__('SPO2')
#         if flag:
#             txts = message.split(' ')
#             heart_rate = int(txts[0][4:])
#             spo2 = int(txts[1][5:])
#             hr_message = getHeartRateType(heart_rate)
#             spo2_message = getSpO2Type(spo2)
#
#             pred_intents = predict_class(hr_message)
#             hr_res = get_response(pred_intents, intents)
#             pred_intents = predict_class(spo2_message)
#             spo2_res = get_response(pred_intents, intents)
#             print(hr_res + '\n' + spo2_res)
#         else:
#             pred_intents = predict_class(message)
#             res = get_response(pred_intents, intents)
#             print(res)
