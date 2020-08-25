"""
Functions/pipeline are/is inspired by this Github Gist
(downloaded 24th of August 2020):
https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
"""

import re
import unicodedata

import contractions
import inflect
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK files
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


def remove_URLs(text: str) -> str:
    """
    Remove URLs from a text
    """
    url_regex = r"(?:(?:http|ftp)s?:\/\/|www\.)[\n\S]+"
    return re.sub(url_regex, "", text)


def replace_contractions(text: str, slang: bool = False) -> str:
    """
    Replace contractions in string of text
    """
    return contractions.fix(text, slang=slang)


def remove_non_ascii(words: list) -> list:
    """
    Remove non-ASCII characters from a list of tokenized words
    """
    new_words = []
    for word in words:
        new_word = (
            unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("utf-8")
        )
        new_words.append(new_word)
    return new_words


def to_lowercase(words: list) -> list:
    """
    Convert all characters to lowercase from list of tokenized words
    """
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: list) -> list:
    """
    Remove punctuation from list of tokenized words
    """
    new_words = []
    for word in words:
        new_word = re.sub(r"[^\w\s]|_", " ", word)

        # Splitting new word on punctuation
        # and adding them separately
        # e.g. out-of-the-box --> out, of, the, box
        for new_word in new_word.split():
            new_words.append(new_word)
    return new_words


def replace_numbers(words: list) -> list:
    """
    Replace all interger occurrences in list of tokenized
    words with textual representation
    """
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word, comma=" ")

            # Splitting new word on space
            # and adding them separately
            # e.g. one hundred and sixteen
            # --> one, hundred, and, sixteen
            for new_word in new_word.split():
                new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words: list, language: str = "english") -> list:
    """
    Remove stop words from list of tokenized words
    """
    new_words = []
    for word in words:
        if word not in stopwords.words(language):
            new_words.append(word)
    return new_words


def lemmatize_words(words: list) -> list:
    """Lemmatize words in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word, word_pos_tag in pos_tag(words):
        word_category = word_pos_tag[0].lower()
        if word_category in ["a", "n", "n"]:
            lemma = lemmatizer.lemmatize(word, word_category)
        else:
            lemma = lemmatizer.lemmatize(word)
        lemmas.append(lemma)
    return lemmas


def normalize(words: list) -> list:
    """
    Normalizes a list of words
    """
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_words(words)
    return words


def preprocess_text(text: str) -> list:
    """
    Preprocesses text using a series of techniques:
    - Removes URLs
    - Replaces contractions
    - Tokenizes text
    - Removes non-ASCII
    - Converts to lower-case
    - Removes punctuation
    - Replaces numbers with textual representation
    - Removes stop words
    """
    # Remove URLs and replace contradictions
    text = remove_URLs(text)
    text = replace_contractions(text)

    # Tokenize text (convert into words)
    words = word_tokenize(text)

    # Normalize text
    words = normalize(words)

    return words
