#!/bin/bash
conda create -n transformer 
conda activate transformer
pip install -r requirements.txt
python src/train.py
