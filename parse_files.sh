#!/bin/bash

rm dataset_ddr/*
python parser.py
./create_index.sh ./dataset_ddr
