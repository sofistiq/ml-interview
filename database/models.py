from .db import db

class Construct(db.Document):
    text = db.StringField(required=True)
    user_rating = db.IntField(required=True,default=0)
