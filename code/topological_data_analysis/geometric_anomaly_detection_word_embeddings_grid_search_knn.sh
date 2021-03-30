#!/bin/bash
screen -dmS gad_word_embeddings_grid_search_knn -L -Logfile gad_word_embeddings_grid_search_knn.logs python geometric_anomaly_detection_word_embeddings_grid_search.py \
--model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--vocab_size 10000 \
--manifold_dimension 2 \
--search_size 10 \
--use_knn_annulus \
--min_annulus_parameter 1 \
--max_annulus_parameter 500 \
--search_params_max_diff 250 \
--num_cpus 19 \
--output_dir data \
--output_filepath_suffix knn
screen -r gad_word_embeddings_grid_search_knn