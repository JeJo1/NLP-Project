import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pickle
import regex as re
import os
import torch

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

def clean(dataset_str):
    brackets = r'\([^ء-ي]*\)'
    digits = r'\d+'
    end = r'\n'
    spaces = r'\s+'
    # dash_space_digits = r'-\s*\d+\s*-|-\s*\d+\s*|\s*\d+\s*-'
    # slashes = r'(?<!(\(|\)|\.|\،)) / (?!(\(|\)|\.|\،))'
    stars = r'\*+'
    english_semicolon = r';'
    english_comma = r','
    long_dash = r'–'
    tilde = r'~'
    backtick = r'`'
    strange_quote = r'“'
    strange_quote2 = r'’'
    ampersand = r'&'
    underscore = r'_'
    plus = r'\+'
    equal = r'='
    misra_l = r'﴾'
    misra_r = r'﴿'
    idk = r'\u200d'
    idk2 = r'\u200f'
    dot = r'…'
    dot_awi = r'\.{2,}'
    single_quote = r'\''
    shadda_skoon = r'[\u0651][\u0652]'

    dataset_str = re.sub(brackets, ' ', dataset_str)
    # dataset_str = re.sub(dash_space_digits, ' ', dataset_str)
    dataset_str = re.sub(digits, ' ', dataset_str)
    dataset_str = re.sub(end, ' ', dataset_str)
    dataset_str = re.sub(stars, ' ', dataset_str)
    dataset_str = re.sub(tilde, ' ', dataset_str)
    dataset_str = re.sub(backtick, ' ', dataset_str)
    dataset_str = re.sub(ampersand, ' ', dataset_str)
    dataset_str = re.sub(underscore, ' ', dataset_str)
    dataset_str = re.sub(plus, ' ', dataset_str)
    dataset_str = re.sub(equal, ' ', dataset_str)
    dataset_str = re.sub(strange_quote, ' ', dataset_str)
    dataset_str = re.sub(strange_quote2, ' ', dataset_str)
    dataset_str = re.sub(dot, ' ', dataset_str)
    dataset_str = re.sub(dot_awi, ' ', dataset_str)
    dataset_str = re.sub(single_quote, ' ', dataset_str)

    # dataset_str = re.sub(slashes, ' ', dataset_str)

    dataset_str = re.sub(english_semicolon, ' ؛ ', dataset_str)
    dataset_str = re.sub(long_dash, ' - ', dataset_str)
    dataset_str = re.sub(misra_l, ' " ', dataset_str)
    dataset_str = re.sub(misra_r, ' " ', dataset_str)
    dataset_str = re.sub(idk, '', dataset_str)
    dataset_str = re.sub(idk2, '', dataset_str)
    dataset_str = re.sub(shadda_skoon, r'\u0651', dataset_str)
    dataset_str = re.sub(english_comma, ' ، ', dataset_str)

    dataset_str = re.sub(r'\s+\.', '.', dataset_str)
    dataset_str = re.sub(r'\.\s+', '.', dataset_str)

    dataset_str = re.sub(spaces, r' ', dataset_str)

    return dataset_str

def Test(test_sentences, model, sentence_encoder, max_len = 500):
    # test_sentences: el text el kbeer bta3 el test (kaza gomla 3ady b dots b 3ak kteer kda)
    # model: el model el hn-predict byh
    # sentence_encoder: el encoder el hn-convert el text le numbers
    test_sentences_clean = clean(test_sentences)
    test_sentences_clean = test_sentences_clean.split('.')
    # split on \n as well, la mlhash lazma
    
    for test in test_sentences_clean:
        if test == '':      # fy gomal fyha . w \n, fa hyb2a fady, bnshelha
            test_sentences_clean.remove(test)
    # split 3la no2ta wa7da msh aktr, w \n 
    # kda kda el clean bysheel el kaza no2ta

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence1 = ''
    sentence2 = ''
    label2 = []
    # arabic_letters = {'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س','ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي'}
    
    for sentence in test_sentences_clean:
        sentence = SOS + sentence + EOS
        for j in range(0, len(sentence), max_len//2):
            current_sentence = sentence[j:j+max_len]
            
            current_sentence, labels = clamp_sentence(current_sentence, np.full((len(current_sentence)), diacritic2id[''], dtype = np.uint8), max_len)
            test_sentence_clean = encode_sentence(current_sentence)
            test_sentence_clean = sentence_encoder.transform(test_sentence_clean.reshape(-1)).reshape(1, -1)
            
            test_sentence_clean = torch.tensor(test_sentence_clean, dtype=torch.int32).to(device)
            outputs = model(test_sentence_clean)
            _, pred = torch.max(outputs.data, 2)
            pred = pred.cpu().numpy().reshape(-1)

            for i in range(len(current_sentence)):
                sentence1 += current_sentence[i]
                sentence1 += id2diacritic[pred[i]]
            sentence1 += '\n'

            
            if j == 0:
                flag = np.array([ord('ء') <= ord(c) <= ord('ي') for c in current_sentence])

                current_sentence = np.array(list(current_sentence))
                pred = pred[flag]
                
            else:
                flag = np.array([ord('ء') <= ord(c) <= ord('ي') for c in current_sentence[max_len//2:]])

                current_sentence = np.array(list(current_sentence[max_len//2:]))
                pred = pred[max_len//2:][flag]
                
            
            current_sentence = current_sentence[flag]
            sentence2 += ''.join(current_sentence)
            label2 += list(pred)

    id2 = np.arange(len(sentence2))
            
    return sentence1[1:-1], sentence2, label2, id2