#!/bin/bash
screen -dmS preprocess_external_word_embeddings -L -Logfile preprocess_external_word_embeddings.logs python preprocess_external_word_embeddings.py \
--raw_data_dir raw_data \
--output_dir data \
--annoy_index_n_trees 500 \
--scann_num_leaves_scaling 5
screen -r preprocess_external_word_embeddings