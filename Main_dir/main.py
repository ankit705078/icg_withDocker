from flask import Flask, render_template, request,redirect,url_for
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation
from keras.layers import concatenate, BatchNormalization, Input
from keras.layers.merge import add
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
# import cv2 #may cause issue in future but working without this too for now
import string
import time
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def preprocess_img(img_path):
    #inception v3 excepts img in 299*299
    img = load_img(img_path, target_size = (299, 299))
    x = img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess_img(image)
    vec = modelfirst.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen = max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

max_length=34
wordtoix=np.load('wordtoix.npy', allow_pickle=True)
wordtoix=wordtoix.item()
ixtoword = np.load('ixtoword.npy', allow_pickle=True)
ixtoword=ixtoword.item()

base_model = InceptionV3(weights='inception_v3_weights.h5')

modelfirst = Model(base_model.input, base_model.layers[-2].output)
model = load_model('my-cap.h5')
app = Flask(__name__,template_folder='Template')
app.secret_key = b'an_5#y2Lkit"F4Q8z\n\xec]/'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predicted', methods=['GET', 'POST'])
def after():
    global model, resnet, vocab, inv_vocab
    if request.files['file1'].filename == '':

        return redirect(url_for('index'))



    img = request.files['file1']
    img.save('static/file.jpg')
    print("=" * 50)
    print("IMAGE SAVED")
    picpth = 'static/file.jpg'
    img = encode(picpth)
    img = img.reshape(1, 2048)
    # print(img.shape)
    cap = greedy_search(img)
    finalcaption=cap.capitalize() + "."

    '''
    references = ['a car is moving very fast and splashing water on sides',
                  'a person is driving a car which splashes water', 'race car is driven by splashing water',
                  'a car is running very fast', 'a car is splashing water on road']
    finalcaption+='\n'+"BLEU SCORES"
    finalcaption+="\n"+str(sentence_bleu(references, cap, weights=(1, 0, 0, 0)))
    finalcaption+="\n"+str(sentence_bleu(references, cap, weights=(0.5, 0.5, 0, 0)))
    finalcaption+="\n"+str(sentence_bleu(references, cap, weights=(0.33, 0.33, 0.33, 0)))
    finalcaption+="\n"+str(sentence_bleu(references, cap, weights=(0.25, 0.25, 0.25, 0.25)))
    '''
    return render_template('predicted.html', data=finalcaption)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
