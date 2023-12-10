import numpy as np
from bs4 import BeautifulSoup
import contractions
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import tensorflow_datasets as tfds
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from keras import Input, Model
from keras.models import load_model


# nltk.download('stopwords')
# nltk.download('punkt')
stopword_list = stopwords.words('english')
tokenizer = wordpunct_tokenize

nlp = spacy.load('en_core_web_sm')

lst = ['no', 'not', 'nor', 'very', 'too', 'ain\'', 'ain\'t']
stopword_list = [s for s in stopword_list if s not in lst]
tokenizer_sentiment = tfds.deprecated.text.SubwordTextEncoder.load_from_file('sentiment')
TOTAL_LEN = 2048
VOCAB_SIZE = tokenizer_sentiment.vocab_size + 2
EMB_SIZE = 128
result = 0


def process_text_data(text,
                      html_tag_remove=True,
                      contraction=True,
                      remove_spcl_char=True,
                      remove_punct=True,
                      remove_newline_char=True,
                      root_word='lemm',
                      remove_stopword=True,
                      remove_extra_space=True,
                      to_lower=True):

		if html_tag_remove:
			text = BeautifulSoup(text, 'html.parser').get_text()
		if contraction:
			text = contractions.fix(text)
		if remove_spcl_char:
			text = re.sub(r'[^a-zA-z0-9.,!?/:;\"\'\s]', '', text)
		if remove_punct:
			text = ''.join([c for c in text if c not in string.punctuation])
		if remove_newline_char:
			text = text.replace('\n', ' ')
		if root_word == 'lemm':
			text = nlp(text)
			text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
		elif root_word == 'stem':
			stemmer = nltk.porter.PorterStemmer()
			text = ' '.join([stemmer.stem(word) for word in text.split()])
		else:
			raise Exception('Root Word is wrong.')
		if remove_stopword:
			tokens = tokenizer(text)
			tokens = [token.strip() for token in tokens]
			tok = [token for token in tokens if token.lower() not in stopword_list]
			text = ' '.join(tok)
		if remove_extra_space:
			text = re.sub(r'^\s*|\s\s*', ' ', text).strip()
		if to_lower:
			text = text.lower()

		return text

def encoding_and_padding(text):
	en_text = [tokenizer_sentiment.vocab_size] + tokenizer_sentiment.encode(text)
	if len(en_text) < TOTAL_LEN:
		en_text = en_text + [0] * (TOTAL_LEN - len(en_text) - 1) + [tokenizer_sentiment.vocab_size + 1]
	else:
		en_text = en_text[:TOTAL_LEN-1] + [tokenizer_sentiment.vocab_size + 1]

	# print(en_text)

	return en_text

def lstm_model():
	inputs = Input((TOTAL_LEN))
	x = Embedding(VOCAB_SIZE, EMB_SIZE)(inputs)
	x = Bidirectional(LSTM(64, return_sequences=True))(x)
	x = Bidirectional(LSTM(64))(x)
	x = Dropout(0.4)(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.3)(x)
	x = Dense(32, activation='relu')(x)
	outputs = Dense(1, activation="sigmoid")(x)

	model = Model(inputs, outputs)

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.set_weights(load_model('model_lstm.h5').get_weights())

	return model


def res(ar):
  if ar > 0.5:
    return 'POSITIVE'
  else:
    return 'NEGATIVE'

def get_text_encode(text):
	encode = encoding_and_padding(process_text_data(text))
	# print(encode)
	return np.expand_dims(np.array(encode), axis=0)

def get_result(model_name, text):
	text_encode = get_text_encode(text)
	print(text_encode)

	if model_name == 'lstm':
		model = lstm_model()
		result = model.predict(text_encode)
		result = res(result)


	return result