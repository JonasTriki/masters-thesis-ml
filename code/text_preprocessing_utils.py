"""
Functions/pipeline are/is inspired by this Github Gist
(downloaded 24th of August 2020):
https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
"""

import re
from string import digits
from typing import List

import contractions
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

# Download NLTK files
nltk.download("punkt")

from num2words import num2words


def remove_urls(text: str) -> str:
    """
    Remove URLs from a text.

    Parameters
    ----------
    text : str
        Text to remove URLs from.

    Returns
    -------
    new_text : str
        New text without URLs.
    """
    url_regex = r"(?:(?:http|ftp)s?:\/\/|www\.)[\n\S]+"
    return re.sub(url_regex, "", text)


def remove_stopwords(words: list, language: str):
    """
    Removes stop words from list of tokenized words.

    Parameters
    ----------
    words : list
        List of tokenized words.
    language : str
        Language of words

    Returns
    -------
    new_words : list
        List of words without stop words.
    """
    new_words = []
    for word in words:
        if word not in stopwords.words(language):
            new_words.append(word)
    return new_words


def replace_contractions(text: str, slang: bool = False) -> str:
    """
    Replace contractions in string of text

    Parameters
    ----------
    text : str
        Text to replace contractions from.

        Example replacements:
        - isn't --> is not
        - don't --> do not
        - I'll --> I will
    slang : bool, optional
        Whether or not to include slang contractions (defaults to False).

    Returns
    -------
    new_text : str
        New text without contractions.
    """
    return contractions.fix(text, slang=slang)


def to_lowercase(words: list) -> list:
    """
    Convert all characters to lowercase from list of tokenized words.

    Parameters
    ----------
    words : list
        List of tokenized words.

    Returns
    -------
    new_words : list
        List of words in lowercase.
    """
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: list) -> list:
    """
    Remove punctuation from list of tokenized words.

    Parameters
    ----------
    words : list
        List of tokenized words.

    Returns
    -------
    new_words : list
        List of new words without punctuations.
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


def remove_digits(words: list):
    """
    Removes digits from list of tokenized words.

    Parameters
    ----------
    words : list
        List of tokenized words.

    Returns
    -------
    new_words : list
        List of words without digits.
    """
    new_words = []
    remove_digits_trans = str.maketrans("", "", digits)
    for word in words:
        new_word = word.translate(remove_digits_trans)
        new_words.append(new_word)
    return new_words


def replace_numbers(words: list, lang: str, ordinal: bool = False) -> list:
    """
    Replaces (ordinal) numbers with its textual representation.

    Parameters
    ----------
    words : list
        List of words.
    lang: str
        Language of words (stripped)
    ordinal : bool, optional
        Whether or not to use ordinal textual representation.

    Returns
    -------
    new_words : list
        List of new words with textual representation of numbers.
    """
    new_words = []
    for word in words:
        if ordinal:
            re_results = re.findall(r"(\d+)(?:st|nd|rd|th)", word)
        else:
            re_results = re.findall(r"\d+", word)
        if len(re_results) > 0:
            number = int(re_results[0])
            number_words = num2words(number, lang=lang, ordinal=ordinal)

            # Remove commas
            number_words = number_words.replace(",", "")

            # Splitting number word on space
            # and adding them separately
            # e.g. one hundred and sixteenth
            # --> one, hundred, and, sixteenth
            for new_word in number_words.split():
                new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def replace_all_numbers(words: list, language: str) -> list:
    """
    Replaces normal and ordinal numbers with its textual representation.

    Parameters
    ----------
    words : list
        List of words.
    language : str
        Language of words

    Returns
    -------
    new_words : list
        List of new words with textual representation of numbers.
    """
    lang = language[:2]  # Extract first two characters (e.g. english --> en)
    words = replace_numbers(words, lang, ordinal=True)
    words = replace_numbers(words, lang)
    return words


def text_to_words(
    text: str, should_replace_contractions: bool = True, language: str = "english"
) -> List[str]:
    """
    Converts text into a list of words. Removes URLs and replaces contractions
    from the original text, before tokenizing into words.

    Parameters
    ----------
    text : str
        Text to process.
    should_replace_contractions: bool
        Whether or not to replace contractions (defaults to True).
    language : str
        Language (defaults to "english").

    Returns
    -------
    words : list
        List of words from the original text.
    """
    # text = remove_urls(text)
    if should_replace_contractions and language == "english":

        # We remove the period character from the text before replacing
        # contractions as a hotfix to the current Github issue:
        # https://github.com/kootenpv/contractions/issues/25
        if text.endswith("."):
            text = replace_contractions(text[:-1]) + "."
        else:
            text = replace_contractions(text)

    # Tokenize text (convert into words)
    words = word_tokenize(text, language)

    return words


def preprocess_words(
    words: List[str],
    language: str = "english",
    should_remove_digits: bool = False,
    should_replace_numbers: bool = True,
    should_remove_stopwords: bool = False,
) -> list:
    """
    Preprocesses list of words using a series of techniques:
    - Converts to lower-case
    - Removes punctuation
    - Replaces numbers with textual representation

    Parameters
    ----------
    words : list of str
        List of words to preprocess.
    language : str
        Language (defaults to "english")
    should_remove_digits : bool
        Whether or not to remove digits from text (defaults to False).
    should_replace_numbers : bool
        Whether or not to replace numbers with textual representation
        (defaults to True). Has no effect if should_remove_digits is set to True.
    should_remove_stopwords : bool
        Whether or not to remove stop words (defaults to False).

    Returns
    -------
    words : list of str
        Preprocessed list of words.
    """
    # Apply a series of techniques to the words
    words = to_lowercase(words)
    words = remove_punctuation(words)
    if should_remove_digits:
        words = remove_digits(words)
    elif should_replace_numbers:
        words = replace_all_numbers(words, language)
    if should_remove_stopwords:
        words = remove_stopwords(words, language)

    return words


def preprocess_text(
    text: str,
    language: str = "english",
    should_replace_contractions: bool = True,
    should_remove_digits: bool = False,
    should_replace_numbers: bool = True,
    should_remove_stopwords: bool = False,
) -> List[str]:
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

    Parameters
    ----------
    text : str
        Text to preprocess.
    language : str
        Language (defaults to "english")
    should_replace_contractions : bool
        Whether or not to replace contractions (defaults to True).
    should_remove_digits : bool
        Whether or not to remove digits from text (defaults to False).
    should_replace_numbers : bool
        Whether or not to replace numbers with textual representation
        (defaults to True). Has no effect if should_remove_digits is set to True.
    should_remove_stopwords : bool
        Whether or not to remove stop words (defaults to False).

    Returns
    -------
    words : list of str
        Preprocessed text split into a list of words.
    """
    # Convert to list of words
    words = text_to_words(
        text=text,
        language=language,
        should_replace_contractions=should_replace_contractions,
    )

    # Process words
    words = preprocess_words(
        words=words,
        language=language,
        should_remove_digits=should_remove_digits,
        should_replace_numbers=should_replace_numbers,
        should_remove_stopwords=should_remove_stopwords,
    )

    return words
