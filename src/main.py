import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pymongo
import json
import random

from flask import Flask, request, jsonify
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick, Tokenizer

app = Flask(__name__)
client = pymongo.MongoClient('mongodb://ri_ml-interview_db')

@app.route('/interview/construct/rate', methods=['POST'])
def rate_construct():
    db = client['interviews']
    collection = db['constructs']
    if 'construct' in request.json and 'user_rating' in request.json:
        result = collection.update({
            'construct': request.json['construct']
        },{
            '$set': {
                'user_rating': request.json['user_rating']
            }
        })
        return True
    else:
        return False

@app.route('/interview/construct', methods=['GET'])
def get_random_construct():
    db = client['interviews']
    collection = db['constructs']
    cursor = collection.aggregate([
        {
            '$match': {
                'user_rating': 0
            }
        },
        {
            '$project': {
                '_id': False,
                'construct': True,
                'user_rating': True,
            }
        },
        {
            '$sample': {
                'size': 1
            }
        }
    ])
    return jsonify(list(cursor).pop())

def train():
    db = client['interviews']
    collection = db['constructs']
    constructs = collection.find()
    dictionary = dict()
    trainingDataRaw = []
    testDataRaw = []
    for construct in constructs:
        for word in text_to_word_sequence(construct['construct']):
            rand = random.randint(1, 6)
            if word in dictionary.keys():
                dictionary[word]['count'] += 1
                # dictionary[word]['rating_sum'] += construct['user_rating']
                dictionary[word]['rating_sum'] += rand
            else:
                dictionary[word] = {
                    'count': 1,
                    'rating_sum': random.randint(1, 6),
                    'rating': 0,
                }
    for word in dictionary:
        dictionary[word]['rating'] = dictionary[word]['rating_sum'] / dictionary[word]['count']
        print(word, dictionary[word])
    # t = Tokenizer()
    # t.fit_on_texts(trainingDataRaw)
    # print(t.word_index)

    # trainingData = np.array(trainingDataRaw)
    # testData = np.array(testDataRaw)

    # model = Sequential()
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
    # model.fit(trainingData, testData, epochs=100, verbose=2)

def initDb():
    db = client['interviews']
    collection = db['constructs']
    with open('src/constructs.json') as f:
        data = json.load(f)
        for element in data:
            collection.update({'construct': element['construct']}, element, True)

def main():
    initDb()
    train()

if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0', debug=True)
