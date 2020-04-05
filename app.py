from flask import Flask, request, Response
from flask_cors import CORS
from os import getenv
from dotenv import load_dotenv
from database.db import initialize_db
from database.models import Construct
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
	train_data = [d['text'] for d in constructs]
	test_data = [d['text'] for d in constructs]

	num_words = 1000
	oov_token = '<UNK>'
	pad_type = 'post'
	trunc_type = 'post'

	tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
	tokenizer.fit_on_texts(train_data)

	word_index = tokenizer.word_index

	train_sequences = tokenizer.texts_to_sequences(train_data)
	maxlen = max([len(x) for x in train_sequences])
	train_padded = pad_sequences(
		train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

	test_sequences = tokenizer.texts_to_sequences(test_data)
	test_padded = pad_sequences(
		test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

	# train_data = np.array(dictionary[len(dictionary) // 2])
	# validation_data = np.array(dictionary[:len(dictionary) // 2])

	embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
	hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
	hub_layer(train_sequences)

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
		x=train_sequences,
		epochs=20,
		verbose=1,
	)

	return '', 200

# rate a given construct
@app.route('/rate', methods=['GET'])
def rate_construct():
	pass

app.run(host='0.0.0.0', debug=getenv('DEVELOPMENT', False))
