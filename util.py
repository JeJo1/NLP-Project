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

diacritic2id = pickle.load(open('diacritic2id.pickle', 'rb'))
id2diacritic = {v: k for k, v in diacritic2id.items()}
PAD = chr(0)
SOS = chr(1)
EOS = chr(2)

def remove_diacritics(text):
    return re.sub(r'[\u064B-\u0652]', '', text)

def extract_sentences(text):
    return [SOS + sentence + EOS for sentence in text.split('.')]

def extract_data_single(sentence):
    no_diacritics = remove_diacritics(sentence)
    labels = np.full((len(no_diacritics)), diacritic2id[''], dtype = np.uint8)
    cur_label = ''
    k = -1
    for i in range(len(sentence)):
        if 0x64B <= ord(sentence[i]) <= 0x652:
            cur_label += sentence[i]
            labels[k] = diacritic2id[cur_label]
        else:
            k += 1
            cur_label = ''

    return no_diacritics, labels

def extract_data(sentences):
    no_diacritics = []
    labels = []
    for sentence in sentences:
        no_diacritics_sentence, labels_sentence = extract_data_single(sentence)
        no_diacritics.append(no_diacritics_sentence)
        labels.append(labels_sentence)
    return no_diacritics, labels

def encode_sentence(sentence):
    sentence_encoded = np.array([ord(c) for c in sentence], dtype = np.uint16)
    return sentence_encoded

def encode_sentences(sentences, labels):
    sentences_encoded = np.zeros((len(sentences), len(sentences[0])), dtype = np.uint16)
    labels_encoded = np.zeros((len(sentences), len(sentences[0])), dtype = np.uint8)
    for i, sentence in enumerate(sentences):
        sentences_encoded[i] = encode_sentence(sentence)
        labels_encoded[i] = labels[i]
    return sentences_encoded, labels_encoded

def extract_features(text, max_len):
    sentences = extract_sentences(text)
    sentences_no_diac, labels = extract_data(sentences)
    sentences_no_diac_clamped, labels_clamped = clamp_sentences(sentences_no_diac, labels, max_len)
    sentences_encoded, labels_encoded = encode_sentences(sentences_no_diac_clamped, labels_clamped)
    return sentences_encoded, labels_encoded

def clamp_sentence(sentence, labels, max_len):
    i = 1
    if max_len >= len(sentence):
        i = len(sentence)
    else: 
        i = sentence.rfind(' ', 0, max_len) + 1
    
    sentence_padded = sentence[:i] + PAD * (max_len - i)    
    labels_padded = np.hstack((labels[:i], np.full((max_len - i), diacritic2id[''], dtype = np.uint8)))
    return sentence_padded, labels_padded

def clamp_sentences(sentences, labels, max_len):
    sentences_clamped = []
    labels_clamped = []
    for i in range(len(sentences)):
        sentence_clamped, sentence_labels_clamped = clamp_sentence(sentences[i], labels[i], max_len)
        sentences_clamped.append(sentence_clamped)
        labels_clamped.append(sentence_labels_clamped)
    return sentences_clamped, labels_clamped

def diacritize_string(sentence_test_str, model, sentence_encoder, max_len):
    sentence = SOS + sentence_test_str + EOS
    sentence_no_diacritics, labels = extract_data_single(sentence)
    sentence_no_diac_clamped, labels_clamped = clamp_sentence(sentence_no_diacritics, labels, max_len)
    sentence, labels_encoded = encode_sentences(sentence_no_diac_clamped, labels_clamped)
    sentence = sentence_encoder.transform(sentence.reshape(-1)).reshape(1, -1)
    pred = model.predict(sentence)

    pred = np.argmax(pred, axis=-1).flatten()

    sentence = ''
    for i in range(len(sentence_no_diacritics)):
        sentence += sentence_no_diacritics[i]
        sentence += id2diacritic[pred[i]]
            
    return sentence