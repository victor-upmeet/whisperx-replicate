from typing import Any

import whisperx

device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

from cog import BasePredictor, Input, Path, BaseModel


class ModelOutput(BaseModel):
    detected_language: str
    segments: Any


class Predictor(BasePredictor):
    def __init__(self):
        self.model = None

    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""

        self.model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    def predict(
        self,
        audio_file: Path = Input(description="Audio file")
    ) -> ModelOutput:
        model = self.model
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # delete model if low on GPU resources
        # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)

        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio_file)
        # diarize_model(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        return ModelOutput(
            segments=result["segments"]
        )

