# FireRedVAD Runtime

## NCNN
We use ncnn to support multiple platform, especially mobile

What’s included
- Stream VAD (frame-by-frame, packed cache)
- Non-stream VAD (whole sequence)
- Non-stream AED (3-class: speech, singing, music)

More details in [NCNN README.md](ncnn/)

## onnxruntime

### Rust

- [Rust ONNX Example](../examples/rust-onnx-example/) — Stream VAD using [wavekat-vad](https://github.com/wavekat/wavekat-vad) with pure Rust Mel filterbank + CMVN preprocessing