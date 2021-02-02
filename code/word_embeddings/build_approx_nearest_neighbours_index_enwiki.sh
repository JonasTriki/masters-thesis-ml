#!/bin/bash
python build_approx_nearest_neighbours_index.py \
--model_training_output_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--vocab_size -1 \
--annoy_index_n_trees 250 \
--output_dir ../output/word2vec_ann_indices \
--output_filepath_suffix jan_2021_annoy_index