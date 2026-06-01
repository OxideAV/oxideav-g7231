//! Criterion benchmarks for the G.723.1 encoder hot paths.
//!
//! Round 203 (depth-mode benchmarks): companion to `benches/decode.rs`
//! and `benches/roundtrip.rs`. The encoder is the analysis-heavy half
//! of G.723.1 — LPC (autocorrelation + Levinson + lag window), LSP
//! conversion + split-VQ search, four-subframe loop with open-loop
//! pitch, closed-loop adaptive-codebook refinement, rate-specific
//! fixed-codebook search (MP-MLQ: per-track greedy, 5/6 pulses;
//! ACELP: 4 pulses on stride-8 tracks with two coordinate-descent
//! passes), joint gain quantisation, and bit-packing. Each scenario
//! pins one rate at one duration so a future optimisation round can
//! A/B-test against a stable baseline.
//!
//! Inputs are synthesised in-bench from a deterministic
//! sum-of-sinusoids generator (180 Hz fundamental + harmonics) so the
//! encoder takes the speech-like pitch path rather than a near-silent
//! shortcut. No `docs/` fixtures or external files are read.
//!
//! Scenarios:
//!
//!   - **encode_mpmlq_voiced_1s**: 33 frames (~1 s) of voiced PCM →
//!     6.3 kbit/s MP-MLQ. Baseline for the per-frame cost of the
//!     MP-MLQ track-greedy fixed-codebook search.
//!   - **encode_acelp_voiced_1s**: same PCM at 5.3 kbit/s ACELP.
//!     Coordinate-descent pulse refinement is the dominant cost
//!     delta vs MP-MLQ.
//!   - **encode_mpmlq_silence_1s**: 33 frames of zeros → MP-MLQ.
//!     Confirms the encoder still runs the full pipeline (rather
//!     than a zero-input shortcut) and gives a floor cost.
//!   - **encode_acelp_voiced_5s**: 5 s of voiced PCM at ACELP. Longer
//!     runs catch any pathological cumulative state growth in the
//!     shadow `SynthesisState`.
//!
//! Run with:
//!     cargo bench -p oxideav-g7231 --bench encode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::Encoder;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, SampleFormat};
use oxideav_g7231::CODEC_ID_STR;

const SAMPLE_RATE: u32 = 8_000;
const FRAME_SAMPLES: usize = 240;

fn make_params(bit_rate: Option<u64>) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(SAMPLE_RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p.bit_rate = bit_rate;
    p
}

fn make_encoder(bit_rate: Option<u64>) -> Box<dyn Encoder> {
    oxideav_g7231::encoder::make_encoder(&make_params(bit_rate)).expect("encoder ctor")
}

/// Voiced synthetic signal — 180 Hz fundamental plus three harmonics,
/// matches the integration-test generator so bench / test numbers stay
/// directly comparable.
fn voiced_pcm(frames: usize) -> Vec<i16> {
    let n = frames * FRAME_SAMPLES;
    let mut out = Vec::with_capacity(n);
    let two_pi = 2.0f32 * std::f32::consts::PI;
    for i in 0..n {
        let t = i as f32 / SAMPLE_RATE as f32;
        let v = (two_pi * 180.0 * t).sin() * 0.50
            + (two_pi * 360.0 * t).sin() * 0.25
            + (two_pi * 720.0 * t).sin() * 0.15
            + (two_pi * 1440.0 * t).sin() * 0.08;
        out.push((v * 20_000.0) as i16);
    }
    out
}

fn audio_frame(samples: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        samples: samples.len() as u32,
        pts: Some(0),
        data: vec![bytes],
    })
}

fn encode_run(pcm: &[i16], bit_rate: Option<u64>) -> usize {
    let mut enc = make_encoder(bit_rate);
    let mut total_packet_bytes = 0usize;
    for chunk in pcm.chunks(FRAME_SAMPLES) {
        let f = audio_frame(chunk);
        enc.send_frame(&f).expect("send_frame");
        while let Ok(pkt) = enc.receive_packet() {
            total_packet_bytes += pkt.data.len();
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        total_packet_bytes += pkt.data.len();
    }
    total_packet_bytes
}

fn bench_encode_mpmlq_voiced_1s(c: &mut Criterion) {
    let pcm = voiced_pcm(33);
    let in_bytes = (pcm.len() * 2) as u64;
    let mut g = c.benchmark_group("encode_mpmlq_voiced_1s");
    g.throughput(Throughput::Bytes(in_bytes));
    g.bench_function(BenchmarkId::from_parameter("mpmlq/voiced/8k/1s"), |b| {
        b.iter(|| {
            let n = encode_run(criterion::black_box(&pcm), Some(6_300));
            criterion::black_box(n);
        });
    });
    g.finish();
}

fn bench_encode_acelp_voiced_1s(c: &mut Criterion) {
    let pcm = voiced_pcm(33);
    let in_bytes = (pcm.len() * 2) as u64;
    let mut g = c.benchmark_group("encode_acelp_voiced_1s");
    g.throughput(Throughput::Bytes(in_bytes));
    g.bench_function(BenchmarkId::from_parameter("acelp/voiced/8k/1s"), |b| {
        b.iter(|| {
            let n = encode_run(criterion::black_box(&pcm), Some(5_300));
            criterion::black_box(n);
        });
    });
    g.finish();
}

fn bench_encode_mpmlq_silence_1s(c: &mut Criterion) {
    let pcm = vec![0i16; 33 * FRAME_SAMPLES];
    let in_bytes = (pcm.len() * 2) as u64;
    let mut g = c.benchmark_group("encode_mpmlq_silence_1s");
    g.throughput(Throughput::Bytes(in_bytes));
    g.bench_function(BenchmarkId::from_parameter("mpmlq/silence/8k/1s"), |b| {
        b.iter(|| {
            let n = encode_run(criterion::black_box(&pcm), Some(6_300));
            criterion::black_box(n);
        });
    });
    g.finish();
}

fn bench_encode_acelp_voiced_5s(c: &mut Criterion) {
    let pcm = voiced_pcm(5 * 33);
    let in_bytes = (pcm.len() * 2) as u64;
    let mut g = c.benchmark_group("encode_acelp_voiced_5s");
    g.throughput(Throughput::Bytes(in_bytes));
    g.bench_function(BenchmarkId::from_parameter("acelp/voiced/8k/5s"), |b| {
        b.iter(|| {
            let n = encode_run(criterion::black_box(&pcm), Some(5_300));
            criterion::black_box(n);
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_encode_mpmlq_voiced_1s,
    bench_encode_acelp_voiced_1s,
    bench_encode_mpmlq_silence_1s,
    bench_encode_acelp_voiced_5s,
);
criterion_main!(benches);
