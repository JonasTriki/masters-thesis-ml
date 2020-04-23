import re
import os
import json
from typing import Union
import h5py
import numpy as np
import pickle
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence, tokenizer_from_json

wnl = WordNetLemmatizer()
eng_stopwords = nltk_stopwords.words('english')
nltk.download('wordnet')
nltk.download('stopwords')

def set_kaggle_env_keys(kaggle_path: str = 'kaggle.json') -> None:
    '''Sets environment keys required by the KaggleApi
    
    Args:
        kaggle_path: Path to the Kaggle json file
    '''
    with open(kaggle_path, 'r') as file:
        kaggle_json = json.load(file)

        # Set OS environment variables
        os.environ['KAGGLE_USERNAME'] = kaggle_json['username']
        os.environ['KAGGLE_KEY'] = kaggle_json['key']

def remove_urls(text: str) -> str:
    '''
    TODO: Docs
    '''
    url_re = r'(?i)(?:(?:(https?|ftp|file):\/\/|www\.|ftp\.)|([\w\-_]+(?:\.|\s*\[dot\]\s*[A-Z\-_]+)+))([A-Z\-\.,@?^=%&amp;:\/~\+#]*[A-Z\-\@?^=%&amp;\/~\+#]){2,6}?'
    return re.sub(url_re, '', text)

def clean_text(text: str) -> str:
    '''
    TODO: Docs
    '''
    
    # Remove URLs
    text = remove_urls(text)
      
    # Lower case and remove punctuation
    text = ' '.join(text_to_word_sequence(text))

    # Remove stop words
    text = ' '.join([t for t in text.split() if t not in eng_stopwords])

    # Apply lemmatization
    text = ' '.join([wnl.lemmatize(t) for t in text.split()])
    
    # Remove numbers and strings that have length 1 or less
    text = ' '.join([t for t in text.split() if not t.isnumeric() and len(t) > 1])
    
    return text

def clean_texts_to_files(texts: Union[list, np.ndarray], text_paths_clean: list):
    '''
    TODO: Docs
    '''
    print('Cleaning texts...')
    for text, clean_text_path in zip(tqdm(texts, unit='text'), text_paths_clean):
        with open(clean_text_path, 'w') as file:
            file.write(clean_text(text))

def read_text_files_gen(file_paths: list, show_progress: bool = False):
    '''
    TODO: Docs
    '''
    for path in tqdm(file_paths, unit='file', disable=not show_progress):
        with open(path, 'r') as file:
            yield file.read()

def create_tokenizer(max_vocab_size: int, texts: Union[list, 'Generator'], save_to_file_path: str = None):
    '''
    TODO: Docs
    '''
    print('Creating vocabulary...')
    tokenizer = Tokenizer(max_vocab_size)
    tokenizer.fit_on_texts(tqdm(texts, unit='text'))

    if not save_to_file_path is None:
        with open(save_to_file_path, 'w') as file:
            file.write(tokenizer.to_json())
    
    return tokenizer

def create_tokenizer_paths(max_vocab_size: int, text_paths: list, save_to_file_path: str = None):
    '''
    TODO: Docs
    '''
    text_gen = read_text_files_gen(text_paths)
    return create_tokenizer_paths(max_vocab_size, text_gen, save_to_file_path)

def read_tokenizer(tokenizer_json_path: str):
    '''
    TODO: Docs
    '''
    print('Reading vocabulary...') 
    with open(tokenizer_json_path, 'r') as file:
        tokenizer = tokenizer_from_json(file.read())
    return tokenizer

def save_pickle(result: any, filename: str) -> None:
    '''Saves to file using Pickle.
    Args:
        result: Result to save
        filename: Where to save the result
    '''
    with open(filename, 'wb') as f:
        pickle.dump(result, f)

def load_pickle(filename: str) -> any:
    '''Loads from file using Pickle.
    Args:
        filename: Where to load the result from
    
    Returns:
        result: Loaded result
    '''
    with open(filename, 'rb') as f:
        result = pickle.load(f)
        
    return result

def text_sequences_to_skipgrams_generator(seq_gen: 'Sequences generator', vocab_size: int, sampling_window_size: int, num_negative_samples: int):
    '''
    TODO: Docs
    '''
    # We use sampling table to randomly downsample words which occur often.
    sampling_table = make_sampling_table(vocab_size + 1)
    for text_seq in seq_gen:

        # By using sampling table, we might get empty skipgram pairs at random.
        # We ensure this does not happend by generating skipgrams until we get a result.
        has_data = False
        while not has_data:

            # Create skipgram pairs
            data_pairs, data_labels = skipgrams(
                text_seq,
                vocab_size,
                sampling_table=sampling_table,
                window_size=sampling_window_size,
                negative_samples=num_negative_samples
            )
            if len(data_pairs) > 0:
                has_data = True

        data_pairs = np.array(data_pairs)
        data_labels = np.array(data_labels).reshape(-1, 1)

        # Yield combined data
        yield data_pairs, data_labels

def save_to_h5_generator(target_path: str, generator: 'Generator object', generator_len: int):
    '''
    Save data from generator to Hierarchical Data Format (HDF)
    '''
    with h5py.File(target_path, 'w') as f:
        # Get init batch (for shapes)
        chunk = next(generator)
        
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk.shape[1:]
        dset = f.create_dataset('data', shape=chunk.shape, maxshape=maxshape,
                                chunks=chunk.shape, dtype=chunk.dtype)

        # Write the first chunk of rows
        dset[:] = chunk
        row_count = chunk.shape[0]

        for _ in range(1, generator_len):
            chunk = next(generator)

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk.shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk

            # Increment the row count
            row_count += chunk.shape[0]