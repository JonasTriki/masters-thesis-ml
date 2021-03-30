#!/bin/bash
screen -dmS postprocess_word2vec_dawiki -L -Logfile postprocess_word2vec_dawiki.logs python postprocess_word2vec_embeddings.py \
--model_training_output_dir ../output/word2vec_training/word2vec_dawiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name dawiki \
--vocab_size -1 \
--annoy_index_n_trees 500 \
--scann_num_leaves_scaling 5
screen -r postprocess_word2vec_dawiki
