import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pymongo
import json

from flask import Flask, request, jsonify

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



def initDb():
    db = client['interviews']
    collection = db['constructs']
    with open('src/constructs.json') as f:
        data = json.load(f)
        for element in data:
            collection.update({'construct': element['construct']}, element, True)

def main():
    initDb()

if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0', debug=True)
