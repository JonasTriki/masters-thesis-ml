#!/bin/bash
screen -dmS prepare_num_word_meanings_supervised_data -L -Logfile prepare_num_word_meanings_supervised_data.logs python prepare_num_word_meanings_supervised_data.py \
--model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--model_name word2vec \
--dataset_name enwiki \
--id_estimation_num_neighbours 25 50 100 150 200 \
--semeval_2010_14_word_senses_filepath ../topological_data_analysis/data/semeval_2010_14_word_senses.joblib \
--tps_neighbourhood_sizes 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 \
--raw_data_dir raw_data \
--output_dir data
screen -r prepare_num_word_meanings_supervised_data