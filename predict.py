from typing import Any

import whisperx

device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

from cog import BasePredictor, Input, Path, BaseModel


class ModelOutput(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        language: str = Input(default=None)
    ) -> ModelOutput:
        asr_options = {
            "temperatures": [0.1],
        }

        model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language, asr_options=asr_options)
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        return ModelOutput(
            segments=result["segments"],
            detected_language=result["language"]
        )