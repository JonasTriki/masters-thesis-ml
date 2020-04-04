from os.path import join as join_path
import json
import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()

'''
spaCy (used for language detection)
-----
To install:
!pip install spacy
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
!pip install spacy-langdetect
'''
import scispacy
import spacy
import en_core_sci_lg # Biomedical word embeddings
from spacy_langdetect import LanguageDetector

class CORD19Data():
    '''
    TODO: Docs
    '''
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # Initialize NLP model
        self.nlp = en_core_sci_lg.load(disable=["tagger", "ner"])
        self.nlp.max_length = 2000000
        self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        self.nlp_words_to_check = 100
    
    def _load_metadata(self) -> pd.DataFrame:
        '''
        TODO: Docs
        '''
        print('Loading metadata...')
        cord_metadata_df = pd.read_csv(join_path(self.data_dir, 'metadata.csv'), dtype={
            'pubmed_id': str,
            'Microsoft Academic Paper ID': str, 
            'doi': str
        })
        print('Done!')
        return cord_metadata_df

    def _parse_json_article(self, article_path: str) -> tuple:
        '''Parses a CORD-19 JSON article

        Args:
            article_path: JSON article path to parse

        Returns:
            TODO
        '''
        with open(article_path, 'r') as file:
            content = json.load(file)

            # Extract information
            paper_id = content['paper_id']
            abstract = []
            body_text = []

            # Abstract
            for item in content['abstract']:
                abstract.append(item['text'])

            # Body text
            for item in content['body_text']:
                body_text.append(item['text'])

            return paper_id, '\n'.join(abstract), '\n'.join(body_text)
    
    def _parse_articles(self) -> pd.DataFrame:
        '''
        TODO: Docs
        '''
        print('Parsing JSON articles...')
        all_cord_article_paths = glob.glob(f'{self.data_dir}/**/*.json', recursive=True)
        
        # Initialize DataFrame dictionary
        cord_articles_dict = {'paper_id': [], 'abstract': [], 'body_text': []}
        for article_path in tqdm(all_cord_article_paths, unit='article'):
            paper_id, abstract, body_text = self._parse_json_article(article_path)
            cord_articles_dict['paper_id'].append(paper_id)
            cord_articles_dict['abstract'].append(abstract)
            cord_articles_dict['body_text'].append(body_text)

        df = pd.DataFrame(cord_articles_dict)
        print('Done!')
        return df
    
    def _merge_metadata_articles(self, metadata_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
        '''
        TODO: Docs
        '''
        print('Merging DataFrames...')
        df = pd.merge(articles_df, metadata_df, left_on='paper_id', right_on='sha', how='left')
        df = df.drop(['sha', 'abstract_y'], axis=1)
        df = df.rename(columns = {'abstract_x': 'abstract', 'source_x': 'source'})
        
        print('Done!')
        return df

    def _exclude_non_metadata_articles(self, df: pd.DataFrame):
        '''
        TODO: Docs
        '''
        print('Excluding articles without metadata...')
        df = df[df.full_text_file.notna()]
        
        print('Done!')
        return df

    def _remove_duplicates(self, df: pd.DataFrame):
        '''
        TODO: Docs
        '''
        print('Removing duplicates...')
        df.drop_duplicates(['abstract', 'body_text'], inplace=True)

        print('Done!')
        return df
    
    def _extract_language(self, text: str) -> str:
        '''
        TODO: Docs
        '''
        # Extract language using spaCy
        text_first_words = ' '.join(text.split(maxsplit=self.nlp_words_to_check)[:self.nlp_words_to_check])
        lang = self.nlp(text_first_words)._.language['language']
        
        return lang
    
    def _perform_lang_detection(self, df: pd.DataFrame):
        '''
        TODO: Docs
        '''
        print('Performing language detection...')
        
        # Extract language
        df['language'] = df.body_text.progress_apply(self._extract_language)

        print('Done!')
        return df
    
    def process_data(self):
        '''Processes the CORD-19 data.
        
        Loads and pre-processes CORD-19 data in specified data directory.
        We take inspiration from/follow Daniel Wolffram's "CORD-19: Create Dataframe" Notebook
        - https://www.kaggle.com/danielwolffram/cord-19-create-dataframe
        
        Returns:
            cord_df: Pandas DataFrame with processed CORD-19 data
        '''
        # Perform pre-processing
        metadata_df = self._load_metadata()
        articles_df = self._parse_articles()
        cord_df = self._merge_metadata_articles(metadata_df, articles_df)
        cord_df = self._exclude_non_metadata_articles(cord_df)
        cord_df = self._remove_duplicates(cord_df)
        cord_df = self._perform_lang_detection(cord_df)

        return cord_df