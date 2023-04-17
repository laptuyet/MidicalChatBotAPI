import json
import pickle
import random
import string

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD


def get_stopwords_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [sw.strip() for sw in stopwords]
        return stopwords


# Load stop words
stop_word_file_path = './vietnamese_stop_words.txt'
stop_words = get_stopwords_list(stop_word_file_path)

# Load intents
intents = json.loads(open('./intents_vn.json', encoding='utf-8').read())

words = []
classes = []
documents = []


def remove_stop_word_from_pattern(sentence):
    for sw in stop_words:
        if ' ' + sw + ' ' in sentence:
            sentence = sentence.replace(sw, '')
    return sentence


# Remove punctuation
def remove_punctuation(sentence):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence


# Remove Vn stopword from patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = remove_punctuation(pattern)
        pattern = remove_stop_word_from_pattern(pattern)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)  # Ví dụ có 6 tag thì [0 0 0 0 0 0]

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [word.lower() for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=8, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done!')
