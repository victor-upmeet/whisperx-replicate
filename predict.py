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
    def __init__(self):
        self.model = None

    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""

        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        temperature: float = Input(
            default=0,
            description="temperature to use for sampling",
        ),
        initial_prompt: str = Input(
            default=None,
            description="optional text to provide as a prompt for the first window.",
        ),
        best_of: int = Input(
            default=5,
            description="number of candidates when sampling with non-zero temperature",
        ),
        no_speech_threshold: float = Input(
            default=0.6,
            description="temperature to use for sampling",
        ),
    ) -> ModelOutput:
        model = self.model
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(
            audio,
            batch_size=batch_size,
            temperature=temperature,
            initial_prompt=initial_prompt,
            best_of=best_of,
            no_speech_threshold=no_speech_threshold
        )

        return ModelOutput(
            segments=result["segments"],
            detected_language=result["language"]
        )

