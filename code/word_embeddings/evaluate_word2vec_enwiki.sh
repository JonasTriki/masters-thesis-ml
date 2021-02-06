#!/bin/bash
python evaluate_word2vec.py \
--model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--sswr_dataset_filepath data/sswr.joblib \
--msr_dataset_filepath data/msr.joblib \
--pad_dataset_filepath data/pad.joblib \
--vocab_size -1 \
--annoy_index_filepath ../output/word2vec_ann_indices/word2vec_enwiki_jan_2021_annoy_index.ann \
--top_n_prediction 1 \
--output_dir ../output/word2vec_eval_analogies
