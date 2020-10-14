#!/bin/bash
python preprocess_wikipedia_data.py \
--language english \
--wiki_dump_time 20200901 \
--raw_data_dir raw_data \
--output_dir data/enwiki-20200901 \
--min_sent_word_count 5 \
--num_output_files -1 \
--max_wikipedia_files -1
