# Rust example using wavekat-vad

Uses [wavekat-vad](https://github.com/wavekat/wavekat-vad), a Rust library that provides a unified interface for multiple VAD backends including FireRedVAD. The FireRedVAD ONNX model and CMVN stats are downloaded and embedded in the binary at compile time — no manual model setup needed.

## Features

- Pure Rust Mel filterbank + CMVN preprocessing (no C dependencies for audio features)
- Automatic resampling from any sample rate to 16kHz
- `FrameAdapter` handles frame buffering (feed any chunk size, get correctly sized 10ms frames)
- Works with any WAV file format (mono/stereo, any sample rate)

## Usage

```sh
cargo run -- /path/to/audio.wav
```

Sample output:

```
File: audio.wav (16000Hz, 1ch, 16bit)
Duration: 3.50s (56000 samples at 16000Hz)

FireRedVAD — frame: 160 samples (10ms)

       0ms  0.000
      10ms  0.000
      20ms  0.012
      30ms  0.008
      40ms  0.245
      50ms  0.876  ###################################  SPEECH
      60ms  0.923  ####################################  SPEECH
     ...
```
