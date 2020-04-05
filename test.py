import json
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import explore_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


def _load_and_shuffle_data(data_path, file_name, cols, seed, separator=',', header=0):
	np.random.seed(seed)
	data_path = os.path.join(data_path, file_name)
	data = pd.read_csv(data_path, usecols=cols, sep=separator, header=header)
	return data.reindex(np.random.permutation(data.index))

def load_dataset(data_path, validation_split=0.2, seed=123):
	columns = (1, 2)
	data = _load_and_shuffle_data(data_path, 'construct.csv', columns, seed)

	# Get the review phrase and sentiment values.
	texts = list(data['text'])
	labels = np.array(data['user_rating'] - 1)
	return _split_training_and_validation_sets(texts, labels, validation_split)


def _split_training_and_validation_sets(texts, labels, validation_split):
	num_training_samples = int((1 - validation_split) * len(texts))
	return ((texts[:num_training_samples], labels[:num_training_samples]),
				 (texts[num_training_samples:], labels[num_training_samples:]))

def ngram_vectorize(train_texts, train_labels, val_texts):
	kwargs = {
		'ngram_range': (1, 2),
		'dtype': 'int32',
		'strip_accents': 'unicode',
		'decode_error': 'replace',
		'analyzer': 'word',
		'min_df': 1,
	}
	vectorizer = TfidfVectorizer(**kwargs)

	x_train = vectorizer.fit_transform(train_texts)
	x_val = vectorizer.transform(val_texts)

	selector = SelectKBest(f_classif, k=min(20000, x_train.shape[1]))
	selector.fit(x_train, train_labels)

	x_train = selector.transform(x_train).astype('float32').toarray()
	x_val = selector.transform(x_val).astype('float32')

	return x_train, x_val


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
	model = models.Sequential()
	model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

	for _ in range(layers - 1):
		model.add(Dense(units=units, activation='relu'))
		model.add(Dropout(rate=dropout_rate))

	model.add(Dense(units=5, activation='softmax'))
	return model


def train_ngram_model(
	data,
	learning_rate=1e-4,
	epochs=1000,
	batch_size=128,
	layers=2,
	units=64,
	dropout_rate=0.2,
):
	(train_texts, train_labels), (val_texts, val_labels) = data

	x_train, x_val = ngram_vectorize(train_texts, train_labels, val_texts)

	model = mlp_model(
		layers=layers,
		units=units,
		dropout_rate=dropout_rate,
		input_shape=x_train.shape[1:],
		num_classes=5,
	)

	optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
	model.compile(
		optimizer=optimizer,
		loss='sparse_categorical_crossentropy',
		metrics=['acc']
	)

	callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

	history = model.fit(
		x_train,
		train_labels,
		epochs=epochs,
		callbacks=callbacks,
		validation_data=(x_val, val_labels),
		verbose=2,
		batch_size=batch_size,
	)

	history = history.history
	print('Validation accuracy: {acc}, loss: {loss}'.format(
		acc=history['val_acc'][-1], loss=history['val_loss'][-1]
	))

	model.save('model.h5')

	return history['val_acc'][-1], history['val_loss'][-1]

dataset = load_dataset('')
train_ngram_model(dataset)
