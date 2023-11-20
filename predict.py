from cog import BasePredictor, Input, Path, BaseModel
from typing import Any

import gc
import os
import shutil
import whisperx
import time
import torch

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"


class ModelOutput(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            language: str = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None),
            initial_prompt: str = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=64),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset",
                default=0.363),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=False),
            alignment_before_diarization: bool = Input(
                description="If align_output and diarization are set to true: set true to diarize at word level, "
                            "set false to diarize at segment level",
                default=True
            ),
            huggingface_access_token: str = Input(
                description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
                            "the user agreement for the models specified in the README.",
                default=None),
            min_speakers: int = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: int = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=False)
    ) -> ModelOutput:
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model("./models/fast-whisper-large-v2", device,
                                        compute_type=compute_type, language=language, asr_options=asr_options,
                                        vad_options=vad_options)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if align_output and diarization:
                if alignment_before_diarization:
                    result = self.align(audio, result, debug)
                    result = self.diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)
                else:
                    result = self.diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)
                    result = self.align(audio, result, debug)
            else:
                if align_output:
                    result = self.align(audio, result, debug)

                if diarization:
                    result = self.diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return ModelOutput(
            segments=result["segments"],
            detected_language=detected_language
        )

    def align(self, audio, result, debug):
        start_time = time.time_ns() / 1e6

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                return_char_alignments=False)

        if debug:
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to align output: {elapsed_time:.2f} ms")

        gc.collect()
        torch.cuda.empty_cache()
        del model_a

        return result

    def diarize(self, audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
        start_time = time.time_ns() / 1e6

        diarize_model = whisperx.DiarizationPipeline(model_name='pyannote/speaker-diarization@2.1',
                                                     use_auth_token=huggingface_access_token, device=device)
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        if debug:
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

        gc.collect()
        torch.cuda.empty_cache()
        del diarize_model

        return result
