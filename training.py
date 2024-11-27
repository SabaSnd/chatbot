import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words: list[str] = []
classes: list[str] = []
documents: list[tuple[list[str], str]] = []

ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# remove duplicates and sort
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

train_x: list[list[int]] = []
train_y: list[list[int]] = []
output_empty = [0] * len(classes)

for document in documents:
    bag: list[int] = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row: list[int] = list(output_empty)
    output_row[classes.index(document[1])] = 1
    train_x.append(bag)
    train_y.append(output_row)   

shuffle_order = list(zip(train_x, train_y))
random.shuffle(shuffle_order)


train_x = list(map(lambda x: x[0], shuffle_order))
train_y = list(map(lambda x: x[1], shuffle_order))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

ask_mod = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('askhole_model.h5', ask_mod)
print("Completed!")