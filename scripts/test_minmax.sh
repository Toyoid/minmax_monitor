#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python -m src.train.minmax_trainer "$@"