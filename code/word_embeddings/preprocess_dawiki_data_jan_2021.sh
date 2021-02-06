#!/bin/bash
screen -dmS preprocess_dawiki_data -L -Logfile preprocess_dawiki_data.logs python preprocess_wikipedia_data.py \
--language danish \
--wiki_name dawiki \
--wiki_dump_time 20210101 \
--raw_data_dir raw_data \
--output_dir data/dawiki-20210101 \
--min_sent_word_count 5 \
--num_output_files -1 \
--max_wikipedia_files -1
