#!/bin/bash
python geometric_anomaly_detection_word_embeddings_grid_search.py \
--model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--vocab_size 10000 \
--annoy_index_filepath ../output/word2vec_ann_indices/word2vec_enwiki_jan_2021_annoy_index.ann \
--manifold_dimension 2 \
--num_radii_to_use 20 \
--max_annulus_radii_diff 0.25 \
--output_dir data