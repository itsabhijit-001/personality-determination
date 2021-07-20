from __future__ import division, print_function
# coding=utf-8
import sys
import os
import re
import numpy as np
# import tensorflow as tf
import neattext.functions as nfx
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Flask utils
from flask import Flask, redirect, url_for, request, render_template,flash
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
app.secret_key = "super secret key"
vectorizer=pickle.load(open('personality_tokenizer.pkl', 'rb'))
model=pickle.load(open('catboost_model.pkl', 'rb'))
target_encoder=pickle.load(open('lbl_target.pkl','rb'))

def encode_to_text(category):
    personality=['yours predicted as ']
    if category[0]=='I':
        personality.append('introvert,')
    else:
        personality.append('extrovert,')
    
    if category[1]=='N':
        personality.append('intuitive,')
    else:
        personality.append('sensing,')
    
    if category[2]=='T':
        personality.append('thinker,')
    else:
        personality.append('emotional,')
    
    if category[3]=='J':
        personality.append('and judging person.')
    else:
        personality.append('and conscious person.')

    return ' '.join(personality)


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def thought_status():
    personality=''
    if request.method == 'POST':
        # check if the post request has the file part
        sentence=str(request.form.get('thought'))
        sentence=sentence.lower()
        sentence=nfx.remove_urls(sentence)
        sentence=nfx.remove_emojis(sentence)
        sentence=nfx.remove_dates(sentence)
        sentence=nfx.remove_special_characters(sentence)
        sentence=nfx.remove_hashtags(sentence)
        sentence=nfx.remove_stopwords(sentence)
        sentence=nfx.remove_numbers(sentence)
        if len(sentence.split())>5:
            sent_vec=vectorizer.transform([sentence]).toarray()
        # Make prediction
            pred=model.predict(sent_vec)[0][0]
            # print(pred)
            # print(target_encoder.inverse_transform([pred]))
            category=target_encoder.inverse_transform([pred])
            # print(category[0])
            personality=encode_to_text(category[0])
            # print(personality)
        else:
            flash('Invalid text please write something meaningful')
            personality='Invalid text please write something meaningful or quite longer.'
    personality=personality.upper()
    return render_template('index.html',personality=personality)

if __name__ == '__main__':
    app.run(debug=True)