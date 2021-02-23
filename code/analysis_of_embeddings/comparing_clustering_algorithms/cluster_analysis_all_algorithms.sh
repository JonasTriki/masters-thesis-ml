#!/bin/bash
screen -dmS cluster_analysis_all_algs_word2vec_enwiki_jan_2021 -L -Logfile cluster_analysis_all_algs_word2vec_enwiki_jan_2021.logs python cluster_analysis_all_algorithms.py \
--model_dir ../../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--vocab_size 10000 \
--output_filepath_suffix cluster_labels \
--output_dir ../../output/word2vec_cluster_analysis
screen -r cluster_analysis_all_algs_word2vec_enwiki_jan_2021
