#!/bin/bash
screen -dmS preprocess_enwiki_data -L -Logfile preprocess_enwiki_data.logs python preprocess_wikipedia_data.py \
--language english \
--wiki_name enwiki \
--wiki_dump_time 20210101 \
--raw_data_dir raw_data \
--output_dir data/enwiki-20210101 \
--min_sent_word_count 5 \
--num_output_files -1 \
--max_wikipedia_files -1
