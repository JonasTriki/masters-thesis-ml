'''
TODO: Add some description
'''
import os
from os.path import join as join_path
from utils import set_kaggle_env_keys, clean_text, text_files_gen, save_pickle, text_sequences_to_skipgrams_generator, save_to_h5_generator
set_kaggle_env_keys()  # Set Kaggle environment keys (KAGGLE_USERNAME, KAGGLE_KEY) before importing the API.

from kaggle.api import KaggleApi
from cord_19_data import CORD19Data
import pandas as pd
import h5py
import numpy as np
rng_seed = 399
np.random.seed(rng_seed)
from tqdm.auto import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from sklearn.model_selection import train_test_split

# Define constants
# ----------------
kaggle_dataset = 'allen-institute-for-ai/CORD-19-research-challenge'
data_dir = 'data'
cord_data_raw_dir = join_path(data_dir, 'raw')
cord_data_out_path = join_path(data_dir, 'cord-19-data.csv')
# cord_data_cleaned_texts_dir = join_path(data_dir, 'cleaned_texts')
cord_data_tokenizer_config_path = join_path(data_dir, 'cord-19-tokenizer.json')
# os.makedirs(cord_data_cleaned_texts_dir, exist_ok=True)

# We have over 30k articles with a large vocubulary. Due to computational
# restrictions, we limit ourselves to the top 10k words from all papers.
max_vocab_size = 10000
sampling_window_size = 5
num_negative_samples = 5
# ----------------

# Download Kaggle dataset
download_cord_dataset = False
if download_cord_dataset:
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(kaggle_dataset, path=cord_data_raw_dir, quiet=False, unzip=True)

# Process raw CORD-19 data into Pandas DataFrame
process_cord_data = False
if process_cord_data:
    cord_df = CORD19Data(cord_data_raw_dir).process_data()
    cord_df.to_csv(cord_data_out_path, index=False)

# Extract texts from Pandas DataFrame and prepare datasets for Word2Vec
prepare_datasets_for_w2v = True
if prepare_datasets_for_w2v:
    cord_df = pd.read_csv(cord_data_out_path)

    # We focus on english articles
    eng_texts = cord_df[cord_df['language'] == 'en'][['cord_uid', 'body_text']].values
    num_articles = len(eng_texts)
    # cleaned_text_paths = np.array([join_path(cord_data_cleaned_texts_dir, f'{cord_uid}.txt') for cord_uid in eng_texts[:, 0]])

    # Split datasets into train/val/test using a 99/1/1 split respectively.
    # Each dataset will have contain skipgram pairs from a unique set of papers
    # to train/evaluate on.
    train_indices, val_indices = train_test_split(np.arange(num_articles), test_size=0.02, random_state=rng_seed)
    val_indices, test_indices = train_test_split(val_indices, test_size=0.5, random_state=rng_seed)

    # Clean texts (lowercase, remove puncuation, etc.) and save to respective dataset directory
    train_text_paths = [join_path(data_dir, 'train', f'{eng_texts[:, 0][idx]}.txt') for idx in train_indices]
    val_text_paths = [join_path(data_dir, 'val', f'{eng_texts[:, 0][idx]}.txt') for idx in val_indices]
    test_text_paths = [join_path(data_dir, 'test', f'{eng_texts[:, 0][idx]}.txt') for idx in test_indices]
    all_text_paths = [*train_text_paths, *val_text_paths, *test_text_paths]
    clean_all_texts = False
    if clean_all_texts:
        for dataset_name, indices, dataset_text_paths in zip(['train', 'val', 'test'], [train_indices, val_indices, test_indices], [train_text_paths, val_text_paths, test_text_paths]):
            print(f'Cleaning and saving texts for {dataset_name}...')
            dataset_texts = eng_texts[:, 1][indices]
            dataset_dir = join_path(data_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            for text_path, text in zip(tqdm(dataset_text_paths, unit='text'), dataset_texts):
                with open(text_path, 'w') as file:
                    file.write(clean_text(text))

    fit_tokenizer = False
    if fit_tokenizer:
        print('Creating vocabulary...')
        tokenizer = Tokenizer(max_vocab_size)
        tokenizer.fit_on_texts(text_files_gen(all_text_paths))
        with open(cord_data_tokenizer_config_path, 'w') as file:
            file.write(tokenizer.to_json())
    else:
        print('Reading vocabulary...')
        with open(cord_data_tokenizer_config_path, 'r') as file:
            tokenizer = tokenizer_from_json(file.read())
    vocab_size = len(tokenizer.word_index)

    # TODO:
    # - Create generator that
    #   (1) reads .txt files from train/val/test,
    #   (2) uses tokenizer to convert from text to sequence,
    #   (3) creates skipgram pairs from text sequences,
    #   (4) serves these skipgram pairs in batches of size batch_size
    #       to model.

    # Clean texts (lowercase, remove puncuation, lemmatization, etc.)
    # clean_all_texts = False
    # if clean_all_texts:
    #     print('Cleaning texts...')
    #     for text, cleaned_text_path in zip(tqdm(eng_texts[:, 1], unit='text'), cleaned_text_paths):
    #         with open(cleaned_text_path, 'w') as file:
    #             file.write(clean_text(text))

    # # Convert texts to integer sequences, generate skipgram pairs and save to file.
    # for dataset_name, indices in zip(['train', 'val', 'test'], [train_indices, val_indices, test_indices]):
    #     print(f'Creating skipgram pairs for {dataset_name}...')
    #     cord_dataset_path = join_path(data_dir, f'cord_{dataset_name}.h5')
    #     dataset_text_paths = cleaned_text_paths[indices]
    #     num_articles_dataset = len(dataset_text_paths)

    #     # Create skipgram pairs generator
    #     seq_gen = tokenizer.texts_to_sequences_generator(text_files_gen(dataset_text_paths))
    #     skipgram_gen = text_sequences_to_skipgrams_generator(
    #         seq_gen,
    #         vocab_size,
    #         sampling_window_size,
    #         num_negative_samples
    #     )

    #     # Save skipgram pairs to file
    #     save_to_h5_generator(cord_dataset_path, skipgram_gen, num_articles_dataset)

    #     break
    # print(len(tokenizer.word_index))
