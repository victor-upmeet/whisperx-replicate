# whisperX on Replicate

This repo is the codebase used to run the following Replicate models:

- [victor-upmeet/whisperx](https://replicate.com/victor-upmeet/whisperx) : if you don't know which model to use, use this one. It uses a low-cost hardware, which suits most cases
- [victor-upmeet/whisperx-a40-large](https://replicate.com/victor-upmeet/whisperx-a40-large) : if you encounter some memory issues with previous models, consider this one. It can happen when dealing with long audio files and performing alignment and/or diarization
- [victor-upmeet/whisperx-a100-80gb](https://replicate.com/victor-upmeet/whisperx-a100-80gb) : if you encounter some memory issues with previous models, consider this one. It can happen when dealing with long audio files and performing alignment and/or diarization.

# Model Information

WhisperX provides fast automatic speech recognition (70x realtime with large-v3) with word-level timestamps and speaker diarization.

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI’s whisper does not natively support batching, but WhisperX does.

Model used is for transcription is large-v3 from faster-whisper.

For more information about WhisperX, including implementation details, see the [WhisperX github repo](https://github.com/m-bain/whisperX).

# Citation

```
@misc{bain2023whisperx,
      title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio}, 
      author={Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
      year={2023},
      eprint={2303.00747},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```