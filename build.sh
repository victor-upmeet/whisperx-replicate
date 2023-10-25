#!/bin/bash

set -e

faster_whisper_model_dir=models/fast-whisper-large-v2

mkdir -p $faster_whisper_model_dir

wget -O $faster_whisper_model_dir/config.json https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/config.json
wget -O $faster_whisper_model_dir/model.bin https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/model.bin
wget -O $faster_whisper_model_dir/tokenizer.json https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/tokenizer.json
wget -O $faster_whisper_model_dir/vocabulary.txt https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/vocabulary.txt

vad_model_dir=models/vad

mkdir -p $vad_model_dir

wget -O $vad_model_dir/whisperx-vad-segmentation.bin $(python ./get_vad_model_url.py)

cog run python