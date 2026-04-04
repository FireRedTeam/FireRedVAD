use wavekat_vad::backends::firered::FireRedVad;
use wavekat_vad::{FrameAdapter, VoiceActivityDetector};

fn main() {
    let audio_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| String::from("recorder.wav"));

    // Open WAV file
    let mut reader = hound::WavReader::open(&audio_path).expect("failed to open WAV file");
    let spec = reader.spec();
    println!(
        "File: {audio_path} ({}Hz, {}ch, {}bit)",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    if spec.sample_format != hound::SampleFormat::Int {
        panic!("Unsupported sample format. Expect Int.");
    }

    // Read samples (first channel only for multi-channel files)
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .step_by(spec.channels as usize)
        .map(|s| s.expect("failed to read sample"))
        .collect();

    // Resample to 16kHz if needed
    let target_rate = 16000;
    let samples = if spec.sample_rate != target_rate {
        println!("Resampling {}Hz -> {}Hz", spec.sample_rate, target_rate);
        use wavekat_vad::preprocessing::AudioResampler;
        let mut resampler =
            AudioResampler::new(spec.sample_rate, target_rate).expect("failed to create resampler");
        resampler.process(&samples)
    } else {
        samples
    };

    let duration_s = samples.len() as f64 / target_rate as f64;
    println!(
        "Duration: {duration_s:.2}s ({} samples at {target_rate}Hz)\n",
        samples.len()
    );

    // Create FireRedVAD — the ONNX model + CMVN stats are embedded in the binary at compile time
    let vad = FireRedVad::new().expect("failed to create FireRedVAD");
    let caps = vad.capabilities();
    println!(
        "FireRedVAD — frame: {} samples ({}ms)\n",
        caps.frame_size, caps.frame_duration_ms
    );

    // FrameAdapter handles automatic frame buffering so you can feed any chunk size
    let mut adapter = FrameAdapter::new(vad);

    // Process in 20ms chunks (arbitrary — the adapter buffers to the required 10ms frame size)
    let chunk_size = target_rate as usize / 50; // 320 samples = 20ms
    let mut time_ms = 0.0;
    let step_ms = chunk_size as f64 * 1000.0 / target_rate as f64;

    for chunk in samples.chunks(chunk_size) {
        let results = adapter.process_all(chunk, target_rate).unwrap();
        for prob in results {
            let bar = "#".repeat((prob * 40.0) as usize);
            let label = if prob > 0.5 { " SPEECH" } else { "" };
            println!("{time_ms:8.0}ms  {prob:.3}  {bar}{label}");
        }
        time_ms += step_ms;
    }

    println!("\nFinished.");
}
