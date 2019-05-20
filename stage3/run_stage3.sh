#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./run.sh [input_location]"
    exit 1
fi
spark-submit \
    --master local[4] \
    script_stage3.py \
    --input $1