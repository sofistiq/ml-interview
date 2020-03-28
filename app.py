from flask import Flask, request, Response
from flask_cors import CORS
from os import getenv
from dotenv import load_dotenv
from database.db import initialize_db
from database.models import Construct

import random

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['MONGODB_SETTINGS'] = {
	'host': getenv('MONGO_SERVER', None)
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
	print(body)
	Construct.objects.get(id=id).update(**body)
	return '', 200


app.run(host='0.0.0.0', debug=getenv('PRODUCTION', False))
