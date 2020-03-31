from flask import Flask, request, Response
from flask_cors import CORS
from os import getenv
from dotenv import load_dotenv
from database.db import initialize_db
from database.models import Construct
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick, Tokenizer

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

construct_dictionary = dict()

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

@app.route('/train', methods=['GET'])
def train():
	constructs = Construct.objects(user_rating__ne=None)
	for construct in constructs:
		for word in text_to_word_sequence(construct['text']):
			if word in construct_dictionary.keys():
				construct_dictionary[word]['rating_count'] += 1
				construct_dictionary[word]['rating_sum'] += construct_dictionary[word]['user_rating']
			else:
				construct_dictionary[word] = {
					'user_rating': construct['user_rating'],
					'rating_count': 1,
					'rating_sum': construct['user_rating'],
				}
	for word in construct_dictionary:
		construct_dictionary[word]['ai_rating'] = construct_dictionary[word]['rating_sum'] / construct_dictionary[word]['rating_count']
	return Response(json.dumps(construct_dictionary), mimetype="application/json", status=200)

@app.route('/rate', methods=['GET'])
def rate_construct():
	body = request.get_json()
	word_arr = text_to_word_sequence(body['text'])
	rating_sum = 0
	rating_count = 0
	for word in word_arr:
		if word in construct_dictionary.keys():
			rating_sum += construct_dictionary[word]['ai_rating']
			rating_count += 1
	if rating_count > 0:
		body['rating'] = rating_sum / rating_count
	else:
		body['rating'] = 'unable to rate construct'
	return Response(json.dumps(body), mimetype="application/json", status=200)

app.run(host='0.0.0.0', debug=getenv('DEVELOPMENT', False))
