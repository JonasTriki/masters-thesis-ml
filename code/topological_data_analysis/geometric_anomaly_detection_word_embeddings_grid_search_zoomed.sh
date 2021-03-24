#!/bin/bash
screen -dmS gad_word_embeddings_grid_search -L -Logfile gad_word_embeddings_grid_search.logs python geometric_anomaly_detection_word_embeddings_grid_search.py \
--model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--vocab_size 10000 \
--annoy_index_filepath ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase/word2vec_enwiki_05_10_weights_annoy_index.ann \
--manifold_dimension 2 \
--num_radii_to_use 10 \
--min_outer_annulus_radius 1.142065 \
--max_outer_annulus_radius 1.446615 \
--num_cpus -1 \
--output_dir data \
--output_filepath_suffix zoomed
screen -r gad_word_embeddings_grid_search

# --use_ripser_plus_plus \
# --num_cpus 10 \
# --max_annulus_radii_diff 0.25 \