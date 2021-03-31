#!/bin/bash
screen -dmS estimate_num_meanings_supervised -L -Logfile estimate_num_meanings_supervised.logs python estimate_num_meanings_supervised.py
screen -r estimate_num_meanings_supervised