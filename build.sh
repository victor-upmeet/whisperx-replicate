#!/bin/bash

model_dir=models/fast-whisper-large-v2

mkdir -p $model_dir

wget -O $model_dir https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/config.json
wget -O $model_dir https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/model.bin
wget -O $model_dir https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/tokenizer.json
wget -O $model_dir https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/vocabulary.txt

cog run python