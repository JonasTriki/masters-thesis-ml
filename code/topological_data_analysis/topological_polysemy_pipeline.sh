#!/bin/bash
screen -dmS topological_polysemy_pipeline -L -Logfile topological_polysemy_pipeline.logs python topological_polysemy_pipeline.py \
--semeval_word_senses_filepath data/semeval_2010_14_word_senses.joblib \
--word2vec_semeval_model_dir ../output/word2vec_training/word2vec_semeval_2010_task_14 \
--word2vec_enwiki_model_dir ../output/word2vec_training/word2vec_enwiki_jan_2021_word2phrase \
--word2vec_google_news_model_dir ../word_embeddings/data/GoogleNews \
--glove_model_dir ../word_embeddings/data/GloVe \
--fasttext_model_dir ../word_embeddings/data/fastText \
--fasttext_tps_model_dir ../word_embeddings/data/fastTextTPS \
--tps_neighbourhood_sizes 10 40 50 60 100 \
--num_top_k_words_frequencies 10000 \
--cyclo_octane_data_filepath custom_data/cyclo-octane.csv \
--henneberg_data_filepath custom_data/henneberg.csv \
--custom_point_cloud_neighbourhood_size 50 \
--output_dir ../output/topological_polysemy_experimentation
screen -r topological_polysemy_pipeline
