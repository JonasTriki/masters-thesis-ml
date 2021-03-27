#!/bin/bash
screen -dmS gad_word_embeddings_grid_search -L -Logfile gad_word_embeddings_grid_search.logs python geometric_anomaly_detection_word_embeddings_grid_search.py \
--model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--vocab_size 10000 \
--annoy_index_filepath ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase/word2vec_enwiki_05_10_weights_annoy_index.ann \
--manifold_dimension 25 \
--search_size 100 \
--min_annulus_parameter 0 \
--max_annulus_parameter -1 \
--search_params_max_diff 0.5 \
--num_cpus -1 \
--output_dir data
screen -r gad_word_embeddings_grid_search

# --use_ripser_plus_plus \
# --num_cpus 10 \