#!/bin/bash

set -e

download() {
  local file_url="$1"
  local destination_path="$2"

  if [ ! -e "$destination_path" ]; then
    wget -O "$destination_path" "$file_url"
  else
      echo "$destination_path already exists. No need to download."
  fi
}

faster_whisper_model_dir=models/fast-whisper-large-v2
mkdir -p $faster_whisper_model_dir

download "https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/config.json" "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/model.bin" "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/tokenizer.json" "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/guillaumekln/faster-whisper-large-v2/resolve/main/vocabulary.txt" "$faster_whisper_model_dir/vocabulary.txt"

pip install -U git+https://github.com/m-bain/whisperx.git

vad_model_dir=models/vad
mkdir -p $vad_model_dir

download $(python3 ./get_vad_model_url.py) "$vad_model_dir/whisperx-vad-segmentation.bin"

if [ ! -e "hg_access_token.txt" ]; then
  echo "hg_access_token.txt not found. Please create it and write your HuggingFace access token (read) in it."
  exit 1
fi

docker secret create hg_access_token hg_access_token.txt

cog run python