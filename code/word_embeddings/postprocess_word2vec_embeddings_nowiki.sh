#!/bin/bash
screen -dmS postprocess_word2vec_nowiki -L -Logfile postprocess_word2vec_nowiki.logs python postprocess_word2vec_embeddings.py \
--model_training_output_dir ../output/word2vec_training/word2vec_nowiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name nowiki \
--vocab_size -1 \
--annoy_index_n_trees 250