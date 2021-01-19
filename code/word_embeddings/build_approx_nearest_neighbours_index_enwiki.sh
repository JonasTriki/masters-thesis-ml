#!/bin/bash
python build_approx_nearest_neighbours_index.py \
--model_training_output_dir ../output/word2vec_training/word2vec_enwiki_sept_2020_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--annoy_index_n_trees 250 \
--output_dir ../output/word2vec_ann_indices