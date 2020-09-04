import numpy as np
from tqdm.auto import tqdm

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
wn_lemmatizer = WordNetLemmatizer()
eng_stopwords = nltk_stopwords.words('english')

def filter_word(word: str, lem_word: str) -> bool:
    '''
    TODO: Docs
    '''
    return word not in eng_stopwords and word != '.' and len(lem_word) > 1

def clean_sents(sents_raw: list, verbose: bool = False) -> list:
    '''
    TODO: Docs
    '''
    if verbose:
        print('Cleaning sentences...')
    sents = []
    for sent in tqdm(sents_raw, unit='sent', disable=not verbose):
        sent_words = []
        for word in word_tokenize(sent):
            lem_word = wn_lemmatizer.lemmatize(word.lower())
            
            # Filter out stopwords, period (.) and lemmatized words that are one character
            if filter_word(word, lem_word):
                sent_words.append(lem_word)
        sents.append(' '.join(sent_words))

    return sents