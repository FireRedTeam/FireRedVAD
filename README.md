<div align="center">
<h1>
FireRedVAD: A SOTA Industrial-Grade
<br>
Voice Activity Detection & Audio Event Detection
</h1>

</div>

[[PyPI]](https://pypi.org/project/firered-vad/)
[[HuggingFace]](https://huggingface.co/FireRedTeam/FireRedVAD)
[[ModelScope]](https://www.modelscope.cn/models/xukaituo/FireRedVAD)


FireRedVAD is a state-of-the-art (SOTA) industrial-grade Voice Activity Detection (VAD) and Audio Event Detection (AED) solution.
FireRedVAD supports non-streaming/streaming VAD and non-streaming AED. It supports speech/singing/music detection in 100+ languages. Non-streaming VAD achieves 97.57% F1 on FLEURS-VAD-102, outperforming Silero-VAD, TEN-VAD, FunASR-VAD and WebRTC-VAD.



## 🔥 News
- [2026.03.03] We release FireRedVAD as a standalone repository, along with model weights and inference code.
- [2026.02.12] We release [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) (FireRedASR2-AED, FireRedVAD, FireRedLID, and FireRedPunc) with model weights and inference code.



## Quick Start

### Install via pip
```bash
pip install firered-vad

# With PyTorch (if not already installed)
pip install firered-vad[gpu]
```

### Setup (from source)
1. Create a clean Python environment:
```bash
$ conda create --name fireredvad python=3.10
$ conda activate fireredvad
$ git clone https://github.com/FireRedTeam/FireRedVAD.git
$ cd FireRedVAD  # or fireredvad
```

2. Install dependencies and set up PATH and PYTHONPATH:
```bash
$ pip install -r requirements.txt
$ export PATH=$PWD/fireredvad/bin/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
```

3. Download models:
```bash
# Download via ModelScope (recommended for users in China)
pip install -U modelscope
modelscope download --model xukaituo/FireRedVAD --local_dir ./pretrained_models/FireRedVAD

# Download via Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
```

4. Convert your audio to **16kHz 16-bit mono PCM** format if needed:
```bash
$ ffmpeg -i <input_audio_path> -ar 16000 -ac 1 -acodec pcm_s16le -f wav <output_wav_path>
```

### Script Usage
```bash
$ cd examples
$ bash inference_vad.sh
$ bash inference_stream_vad.sh
$ bash inference_aed.sh
```


### Command-line Usage
If installed via `pip install firered-vad`, use the `fireredvad` CLI directly:
```bash
$ fireredvad --task vad --wav_path assets/hello_zh.wav --model_dir pretrained_models/FireRedVAD/VAD
# 2026-03-06 17:30:00,086 (param:16) INFO: #param of DetectModel <class 'fireredvad.core.detect_model.DetectModel'> is 588417 = 2.2 MB (float32)
# 2026-03-06 17:30:00,131 (fireredvad_cli:37) INFO: Result: {'dur': 2.32, 'timestamps': [(0.44, 1.82)], 'wav_path': 'assets/hello_zh.wav'}
$ fireredvad --task stream_vad --wav_path assets/hello_en.wav --model_dir pretrained_models/FireRedVAD/Stream-VAD
# 2026-03-06 17:30:28,318 (param:16) INFO: #param of DetectModel <class 'fireredvad.core.detect_model.DetectModel'> is 567937 = 2.2 MB (float32)
# 2026-03-06 17:30:28,342 (fireredvad_cli:37) INFO: Result: ([StreamVadFrameResult(frame_idx=1, is_speech=0, raw_prob=0.037, smoothed_prob=0.037, is_speech_start=False, is_speech_end=False, speech_start_frame=-1, speech_end_frame=-1), StreamVadFrameResult(frame_idx=2, is_speech=0, raw_prob=0.037, smoothed_prob=0.037, is_speech_start=False, is_speech_end=False, speech_start_frame=-1, speech_end_frame=-1), ...]
$ fireredvad --task aed --wav_path assets/event.wav --model_dir pretrained_models/FireRedVAD/AED
# 2026-03-06 17:31:13,912 (param:16) INFO: #param of DetectModel <class 'fireredvad.core.detect_model.DetectModel'> is 588931 = 2.2 MB (float32)
# 2026-03-06 17:31:13,989 (fireredvad_cli:37) INFO: Result: {'dur': 22.016, 'event2timestamps': {'speech': [(0.4, 3.56), (3.66, 9.08), (9.27, 9.77), (10.78, 21.76)], 'singing': [(1.79, 19.96), (19.97, 22.016)], 'music': [(0.09, 12.32), (12.33, 22.016)]}, 'event2ratio': {'speech': 0.848, 'singing': 0.905, 'music': 0.991}, 'wav_path': 'assets/event.wav'}
```

For source installation, set up `PATH` and `PYTHONPATH` first: `export PATH=$PWD/fireredvad/bin/:$PATH; export PYTHONPATH=$PWD/:$PYTHONPATH`

```bash
$ vad.py --help
$ vad.py --use_gpu 0 --model_dir pretrained_models/FireRedVAD/VAD --smooth_window_size 5 --speech_threshold 0.4 \
    --min_speech_frame 20 --max_speech_frame 2000 --min_silence_frame 20 --merge_silence_frame 0 \
    --extend_speech_frame 0 --chunk_max_frame 30000 --write_textgrid 1 \
    --wav_path assets/hello_zh.wav --output out/vad.txt --save_segment_dir out/vad

$ stream_vad.py --help
$ stream_vad.py --use_gpu 0 --model_dir pretrained_models/FireRedVAD/Stream-VAD --smooth_window_size 5 --speech_threshold 0.4 \
    --pad_start_frame 5 --min_speech_frame 8 --max_speech_frame 2000 --min_silence_frame 20 \
    --chunk_max_frame 30000 --write_textgrid 1 \
    --wav_path assets/hello_en.wav --output out/vad.txt --save_segment_dir out/stream_vad

$ aed.py --help
$ aed.py --use_gpu 0 --model_dir pretrained_models/FireRedVAD/AED --smooth_window_size 5 --speech_threshold 0.4 \
    --singing_threshold 0.5 --music_threshold 0.5 --min_event_frame 20 --max_event_frame 2000 \
    --min_silence_frame 20 --merge_silence_frame 0 --extend_speech_frame 0 --chunk_max_frame 30000 --write_textgrid 1 \
    --wav_path assets/event.wav --output out/aed.txt --save_segment_dir out/aed
```


### Python API Usage
> If installed via `pip install firered-vad`, no `PYTHONPATH` setup is needed.
> For source installation, set up `PYTHONPATH` first: `export PYTHONPATH=$PWD/:$PYTHONPATH`

#### Non-streaming VAD
```python
from fireredvad import FireRedVad, FireRedVadConfig

vad_config = FireRedVadConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    min_speech_frame=20,
    max_speech_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000)
vad = FireRedVad.from_pretrained("pretrained_models/FireRedVAD/VAD", vad_config)

result, probs = vad.detect("assets/hello_zh.wav")

print(result)
# {'dur': 2.32, 'timestamps': [(0.44, 1.82)], 'wav_path': 'assets/hello_zh.wav'}
```


#### Streaming VAD

```python
from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

vad_config=FireRedStreamVadConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    pad_start_frame=5,
    min_speech_frame=8,
    max_speech_frame=2000,
    min_silence_frame=20,
    chunk_max_frame=30000)
stream_vad = FireRedStreamVad.from_pretrained("pretrained_models/FireRedVAD/Stream-VAD", vad_config)

frame_results, result = stream_vad.detect_full("assets/hello_en.wav")

print(result)
# {'dur': 2.24, 'timestamps': [(0.28, 1.83)], 'wav_path': 'assets/hello_en.wav'}
```


#### Non-streaming AED

```python
from fireredvad import FireRedAed, FireRedAedConfig

aed_config=FireRedAedConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    singing_threshold=0.5,
    music_threshold=0.5,
    min_event_frame=20,
    max_event_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000)
aed = FireRedAed.from_pretrained("pretrained_models/FireRedVAD/AED", aed_config)

result, probs = aed.detect("assets/event.wav")

print(result)
# {'dur': 22.016, 'event2timestamps': {'speech': [(0.4, 3.56), (3.66, 9.08), (9.27, 9.77), (10.78, 21.76)], 'singing': [(1.79, 19.96), (19.97, 22.016)], 'music': [(0.09, 12.32), (12.33, 22.016)]}, 'event2ratio': {'speech': 0.848, 'singing': 0.905, 'music': 0.991}, 'wav_path': 'assets/event.wav'}
```



## Method
DFSMN-based non-streaming/streaming Voice Activity Detection and Audio Event Detection.



## Evaluation
### FireRedVAD
We evaluate FireRedVAD on FLEURS-VAD-102, a multilingual VAD benchmark covering 102 languages.

FireRedVAD achieves SOTA performance, outperforming Silero-VAD, TEN-VAD, FunASR-VAD, and WebRTC-VAD.

|Metric\Model|FireRedVAD|Silero-VAD|TEN-VAD|FunASR-VAD|WebRTC-VAD|
|:-------:|:-----:|:------:|:------:|:------:|:------:|
|AUC-ROC↑  |**99.60**|97.99|97.81|-    |-    |
|F1 score↑ |**97.57**|95.95|95.19|90.91|52.30|
|False Alarm Rate↓  |**2.69** |9.41 |15.47|44.03|2.83 |
|Miss Rate↓|3.62     |3.95 |2.95 |0.42 |64.15|

<sup>*</sup>FLEURS-VAD-102: We randomly selected ~100 audio files per language from [FLEURS test set](https://huggingface.co/datasets/google/fleurs), resulting in 9,443 audio files with manually annotated binary VAD labels (speech=1, silence=0). This VAD testset will be open sourced (coming soon).

Note: FunASR-VAD achieves low Miss Rate but at the cost of high False Alarm Rate (44.03%), indicating over-prediction of speech segments.



## FAQ
**Q: What audio format is supported?**

16kHz 16-bit mono PCM wav. Use ffmpeg to convert other formats: `ffmpeg -i <input_audio_path> -ar 16000 -ac 1 -acodec pcm_s16le -f wav <output_wav_path>`
