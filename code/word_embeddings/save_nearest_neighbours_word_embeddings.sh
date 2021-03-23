#!/bin/bash
screen -dmS save_nearest_neighbours_word_embeddings -L -Logfile save_nearest_neighbours_word_embeddings.logs python save_nearest_neighbours_word_embeddings.py \
--model_training_output_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--vocab_size 10000 \
--num_nearest_neighbours 100 \
--num_cpus -1
screen -r save_nearest_neighbours_word_embeddings