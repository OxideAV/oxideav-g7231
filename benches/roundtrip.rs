//! Criterion benchmark for the full G.723.1 encode → decode round-trip.
//!
//! Round 203 (depth-mode benchmarks): pair with `benches/encode.rs`
//! and `benches/decode.rs`. This file deliberately measures the
//! end-to-end trait-surface path (i.e. the cost a real OxideAV
//! pipeline sees when transcoding one second of speech) rather than
//! either half in isolation. Encoder + decoder both reuse their
//! internal post-quant tables / `SynthesisState` ports of one
//! another, so the round-trip number is the canonical reference for
//! "what does it cost to ship G.723.1 in a pipeline".
//!
//! All inputs are synthesised in-bench from a deterministic
//! sum-of-sinusoids generator. No `docs/` fixtures or external files
//! are read.
//!
//! Scenarios:
//!
//!   - **roundtrip_mpmlq_voiced_1s**: 33 frames of voiced PCM →
//!     6.3 kbit/s MP-MLQ encode → registered full-synthesis decoder.
//!   - **roundtrip_acelp_voiced_1s**: same input at 5.3 kbit/s ACELP.
//!   - **roundtrip_mpmlq_voiced_5s**: longer voiced run to surface
//!     any cumulative cost in either half (e.g. allocator churn in
//!     the shadow `SynthesisState`).
//!
//! Run with:
//!     cargo bench -p oxideav-g7231 --bench roundtrip

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, SampleFormat};
use oxideav_core::{CodecRegistry, Decoder, Encoder};
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

fn make_decoder() -> Box<dyn Decoder> {
    let mut reg = CodecRegistry::new();
    oxideav_g7231::register_codecs(&mut reg);
    reg.first_decoder(&make_params(None)).expect("decoder ctor")
}

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

fn roundtrip(pcm: &[i16], bit_rate: Option<u64>) -> usize {
    let mut enc = make_encoder(bit_rate);
    let mut dec = make_decoder();
    let mut decoded_samples = 0usize;
    for chunk in pcm.chunks(FRAME_SAMPLES) {
        let f = audio_frame(chunk);
        enc.send_frame(&f).expect("send_frame");
        while let Ok(pkt) = enc.receive_packet() {
            dec.send_packet(&pkt).expect("send_packet");
            while let Ok(frame) = dec.receive_frame() {
                if let Frame::Audio(af) = frame {
                    decoded_samples += af.samples as usize;
                }
            }
        }
    }
    enc.flush().expect("flush enc");
    while let Ok(pkt) = enc.receive_packet() {
        dec.send_packet(&pkt).expect("send_packet");
    }
    dec.flush().expect("flush dec");
    while let Ok(frame) = dec.receive_frame() {
        if let Frame::Audio(af) = frame {
            decoded_samples += af.samples as usize;
        }
    }
    decoded_samples
}

fn bench_roundtrip_mpmlq_voiced_1s(c: &mut Criterion) {
    let pcm = voiced_pcm(33);
    let in_bytes = (pcm.len() * 2) as u64;
    let mut g = c.benchmark_group("roundtrip_mpmlq_voiced_1s");
    g.throughput(Throughput::Bytes(in_bytes));
    g.bench_function(BenchmarkId::from_parameter("mpmlq/voiced/8k/1s"), |b| {
        b.iter(|| {
            let n = roundtrip(criterion::black_box(&pcm), Some(6_300));
            criterion::black_box(n);
        });
    });
    g.finish();
}

fn bench_roundtrip_acelp_voiced_1s(c: &mut Criterion) {
    let pcm = voiced_pcm(33);
    let in_bytes = (pcm.len() * 2) as u64;
    let mut g = c.benchmark_group("roundtrip_acelp_voiced_1s");
    g.throughput(Throughput::Bytes(in_bytes));
    g.bench_function(BenchmarkId::from_parameter("acelp/voiced/8k/1s"), |b| {
        b.iter(|| {
            let n = roundtrip(criterion::black_box(&pcm), Some(5_300));
            criterion::black_box(n);
        });
    });
    g.finish();
}

fn bench_roundtrip_mpmlq_voiced_5s(c: &mut Criterion) {
    let pcm = voiced_pcm(5 * 33);
    let in_bytes = (pcm.len() * 2) as u64;
    let mut g = c.benchmark_group("roundtrip_mpmlq_voiced_5s");
    g.throughput(Throughput::Bytes(in_bytes));
    g.bench_function(BenchmarkId::from_parameter("mpmlq/voiced/8k/5s"), |b| {
        b.iter(|| {
            let n = roundtrip(criterion::black_box(&pcm), Some(6_300));
            criterion::black_box(n);
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_roundtrip_mpmlq_voiced_1s,
    bench_roundtrip_acelp_voiced_1s,
    bench_roundtrip_mpmlq_voiced_5s,
);
criterion_main!(benches);
