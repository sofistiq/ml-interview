from flask import Flask, request, Response
from flask_cors import CORS
from os import getenv
from dotenv import load_dotenv
from database.db import initialize_db
from database.models import Construct
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick, Tokenizer

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import random
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['MONGODB_SETTINGS'] = {
	'host': getenv('MONGO_HOST', None),
	'db': getenv('MONGO_DATABASE', None),
}

initialize_db(app)

@app.route('/construct', methods=['GET'])
def get_random_construct():
	construct = random.choice(Construct.objects(user_rating=None)).to_json()
	return Response(construct, mimetype="application/json", status=200)

@app.route('/construct/<id>', methods=['GET'])
def get_construct_by_id(id):
	construct = random.choice(Construct.objects(id=id)).to_json()
	return Response(construct, mimetype="application/json", status=200)

@app.route('/construct/<id>', methods=['PATCH'])
def update_construct(id):
	body = request.get_json()
	Construct.objects.get(id=id).update(**body)
	return '', 200

# train using existing dataset
@app.route('/train', methods=['GET'])
def train():
	constructs = Construct.objects(user_rating__ne=None)
	dictionary = np.array([d['text'] for d in constructs])
	train_data = np.array(dictionary[len(dictionary) // 2])
	validation_data = np.array(dictionary[:len(dictionary) // 2])
	embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
	hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
	hub_layer(dictionary)

	model = tf.keras.Sequential()
	model.add(hub_layer)
	model.add(tf.keras.layers.Dense(16, activation='relu'))
	model.add(tf.keras.layers.Dense(1))

	model.summary()

	model.compile(
		optimizer='adam',
		loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	history = model.fit(
		train_data,
		epochs=20,
		validation_data=validation_data,
		verbose=1,
	)

	return '', 200

# rate a given construct
@app.route('/rate', methods=['GET'])
def rate_construct():
	pass

app.run(host='0.0.0.0', debug=getenv('DEVELOPMENT', False))
