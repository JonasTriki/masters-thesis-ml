#!/bin/bash
screen -dmS build_ann_index_nowiki -L -Logfile build_ann_index_nowiki.logs python build_approx_nearest_neighbours_index.py \
--model_training_output_dir ../output/word2vec_training/word2vec_nowiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name nowiki \
--vocab_size -1 \
--annoy_index_n_trees 250 \
--output_dir ../output/word2vec_ann_indices \
--output_filepath_suffix jan_2021_annoy_index