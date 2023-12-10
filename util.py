import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pickle
import regex as re
import nltk
import os

tf.keras.backend.clear_session()

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def extract_data(text, no_diacritics_dumpfile = None, labels_dumpfile = None):
    no_diacritics = remove_diacritics(text)
    labels = np.zeros((len(no_diacritics), 2)).astype(np.int16)
    j = -1
    k = 0
    for i in range(len(text)):
        if 0x64B <= ord(text[i]) <= 0x652:
            labels[j, k] = ord(text[i])
            k += 1
        else:
            j += 1
            k = 0
    if no_diacritics_dumpfile is not None:
        with open(no_diacritics_dumpfile, 'wb') as f:
            pickle.dump(no_diacritics, f)
    if labels_dumpfile is not None:
        with open(labels_dumpfile, 'wb') as f:
            pickle.dump(labels, f)
    return no_diacritics, labels

def remove_diacritics(text):
    return re.sub(r'[\u064B-\u0652]', '', text)

def extract_sentences(text, labels, max_len):
    sentences_list = text.split('.')

    sentences = np.zeros((len(sentences_list), max_len)).astype(np.int16)
    diacritics = np.zeros((len(sentences_list), max_len, 2)).astype(np.int16)

    j = 0
    for i, sentence in enumerate(sentences_list):
        last_space = sentence.rfind(' ', 0, max_len) + 1

        if last_space == -1: 
            j += len(sentence) + 1
            continue

        sentences[i, :last_space] = np.array([ord(char) for char in sentence[:last_space]], dtype=np.uint16)
        diacritics[i, :last_space] = labels[j : j + last_space]

        j += len(sentence) + 1
    
    return sentences, diacritics

def diacritize_string(sentence_str, model, sentence_encoder, label_encoder, label_encoder2, max_len):
    sentence = np.array([ord(char) for char in sentence_str], dtype=np.uint16)
    sentence = sentence_encoder.transform(sentence)

    if len(sentence) > max_len:
        sentence = sentence[:max_len]
    else:
        sentence = np.pad(sentence, (0, max_len - len(sentence)), constant_values=0)

    sentence = sentence.reshape(1, max_len)

    pred = model.predict(sentence)
    pred = np.argmax(pred, axis=-1).flatten()
    pred = label_encoder2.inverse_transform(pred)
    pred1 = pred // 9
    pred2 = pred % 9
    pred1 = label_encoder.inverse_transform(pred1)
    pred2 = label_encoder.inverse_transform(pred2)

    sentence = ''
    for i in range(len(sentence_str)):
        sentence += sentence_str[i]
        if pred1[i] != 0:
            sentence += chr(pred1[i])
        if pred2[i] != 0:
            sentence += chr(pred2[i])
            
    return sentence