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

faster_whisper_model_dir=models/faster-whisper-large-v3
mkdir -p $faster_whisper_model_dir

download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json" "$faster_whisper_model_dir/config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin" "$faster_whisper_model_dir/model.bin"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/preprocessor_config.json" "$faster_whisper_model_dir/preprocessor_config.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json" "$faster_whisper_model_dir/tokenizer.json"
download "https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json" "$faster_whisper_model_dir/vocabulary.json"

pip install -U git+https://github.com/m-bain/whisperx.git

vad_model_dir=models/vad
mkdir -p $vad_model_dir

download $(python3 ./get_vad_model_url.py) "$vad_model_dir/whisperx-vad-segmentation.bin"

cog run python