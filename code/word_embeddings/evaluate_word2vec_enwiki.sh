#!/bin/bash
screen -dmS eval_word2vec_enwiki -L -Logfile eval_word2vec_enwiki.logs python evaluate_word2vec.py \
--model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--sswr_dataset_filepath data/sswr.joblib \
--msr_dataset_filepath data/msr.joblib \
--pad_dataset_filepath data/pad.joblib \
--vocab_size -1 \
--approx_nn_path ../output/word2vec_ann_indices/word2vec_enwiki_jan_2021_scann_artifacts \
--approx_nn_alg scann \
--top_n_prediction 1 \
--output_dir ../output/word2vec_eval_analogies
