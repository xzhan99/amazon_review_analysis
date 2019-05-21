#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./run.sh [input_location] [selected_product]"
    exit 1
fi
spark-submit \
    --master local[4] \
    script_stage3.py \
    --input $1
    --selected_product $2