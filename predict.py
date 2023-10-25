from typing import Any

import os
import shutil
import whisperx
import time

device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

from cog import BasePredictor, Input, Path, BaseModel


class ModelOutput(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        language: str = Input(default=None)
    ) -> ModelOutput:
        asr_options = {
            "temperatures": [0.1],
        }

        start_time = time.time_ns()  / 1e6

        model = whisperx.load_model("./models/fast-whisper-large-v2", device, compute_type=compute_type, language=language, asr_options=asr_options)

        elapsed_time = time.time_ns()  / 1e6 - start_time
        print(f"Duration to load model: {elapsed_time:.2f} ms")

        start_time = time.time_ns() / 1e6

        audio = whisperx.load_audio(audio_file)

        elapsed_time = time.time_ns()  / 1e6 - start_time
        print(f"Duration to load audio: {elapsed_time:.2f} ms")

        start_time = time.time_ns() / 1e6

        result = model.transcribe(audio, batch_size=batch_size)

        elapsed_time = time.time_ns()  / 1e6 - start_time
        print(f"Duration to transcribe: {elapsed_time:.2f} ms")

        return ModelOutput(
            segments=result["segments"],
            detected_language=result["language"]
        )