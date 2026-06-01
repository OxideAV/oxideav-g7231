//! Criterion benchmarks for the G.723.1 decoder hot paths.
//!
//! Round 203 (depth-mode benchmarks): pair with `benches/encode.rs` and
//! `benches/roundtrip.rs`. The decoder is the more LUT-heavy half of
//! G.723.1 (LSP dequant lookup, ACB / FCB excitation expansion, LPC
//! synthesis filter, pitch + formant post-filter, tilt compensation,
//! AGC) so its cost shape is genuinely different from the encoder's
//! analysis-by-synthesis pipeline.
//!
//! Each scenario is self-contained: the input bitstream is produced
//! in-bench by running this crate's own encoder against a deterministic
//! synthetic signal. No `docs/` fixtures or external files are read.
//!
//! Scenarios:
//!
//!   - **decode_mpmlq_synth_1s**: ~33 high-rate (24-byte) packets — one
//!     second of speech-like input at 8 kHz — fed through the registered
//!     full-synthesis decoder. Measures the steady-state per-frame cost
//!     of MP-MLQ excitation expansion plus post-filtering.
//!   - **decode_acelp_synth_1s**: same one-second input, encoded at
//!     5.3 kbit/s (20-byte packets). ACELP excitation is built from
//!     4 fixed-codebook pulses; the rest of the decode chain matches
//!     MP-MLQ, so the per-rate delta isolates the FCB expansion cost.
//!   - **decode_erased_5s**: 167 erasure (`0b11`) packets back-to-back.
//!     Drives the comfort-noise / extrapolation path
//!     ([`SynthesisState::decode_erased`]) at sustained load. Useful for
//!     catching any future regression in the gain-attenuation schedule
//!     or the pseudo-random innovation seed cost.
//!   - **decode_mixed_5s**: alternating MP-MLQ / ACELP / erased / SID
//!     packets at 5 s total to exercise the rate-discriminator dispatch
//!     branch as well as state transitions between the three frame
//!     types in the synthesiser.
//!
//! Run with:
//!     cargo bench -p oxideav-g7231 --bench decode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_core::packet::PacketFlags;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, Packet, Rational, SampleFormat, TimeBase,
};
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

/// Deterministic voiced synthetic signal — 180 Hz fundamental plus
/// three harmonics, amplitude scaled so the encoder exercises real
/// pitch + fixed-codebook work rather than near-silent shortcuts.
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

fn packet(data: Vec<u8>) -> Packet {
    Packet {
        stream_index: 0,
        time_base: TimeBase(Rational::new(1, SAMPLE_RATE as i64)),
        pts: None,
        dts: None,
        duration: None,
        flags: PacketFlags::default(),
        data,
    }
}

/// Encode `seconds * 33` whole 30 ms frames worth of voiced PCM at the
/// requested rate, returning the resulting per-frame packets. Done
/// once outside the benchmark loop so the bench measures pure decode
/// cost.
fn encoded_voiced_packets(seconds: usize, bit_rate: Option<u64>) -> Vec<Packet> {
    // 1 s of 30 ms frames = ~33.33 frames; round to 33 so the bench
    // shape is identical across rates and seeds.
    let frames = seconds * 33;
    let pcm = voiced_pcm(frames);
    let mut enc = make_encoder(bit_rate);
    let mut packets = Vec::with_capacity(frames);
    for chunk in pcm.chunks(FRAME_SAMPLES) {
        let f = audio_frame(chunk);
        enc.send_frame(&f).expect("send_frame");
        while let Ok(pkt) = enc.receive_packet() {
            packets.push(pkt);
        }
    }
    enc.flush().expect("flush");
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }
    packets
}

fn bench_decode_mpmlq_synth_1s(c: &mut Criterion) {
    let packets = encoded_voiced_packets(1, Some(6_300));
    let out_bytes = (packets.len() * FRAME_SAMPLES * 2) as u64;
    let mut g = c.benchmark_group("decode_mpmlq_synth_1s");
    g.throughput(Throughput::Bytes(out_bytes));
    g.bench_function(BenchmarkId::from_parameter("mpmlq/8k/1s"), |b| {
        b.iter(|| {
            let mut dec = make_decoder();
            for pkt in &packets {
                dec.send_packet(pkt).expect("send_packet");
                while let Ok(frame) = dec.receive_frame() {
                    criterion::black_box(frame);
                }
            }
        });
    });
    g.finish();
}

fn bench_decode_acelp_synth_1s(c: &mut Criterion) {
    let packets = encoded_voiced_packets(1, Some(5_300));
    let out_bytes = (packets.len() * FRAME_SAMPLES * 2) as u64;
    let mut g = c.benchmark_group("decode_acelp_synth_1s");
    g.throughput(Throughput::Bytes(out_bytes));
    g.bench_function(BenchmarkId::from_parameter("acelp/8k/1s"), |b| {
        b.iter(|| {
            let mut dec = make_decoder();
            for pkt in &packets {
                dec.send_packet(pkt).expect("send_packet");
                while let Ok(frame) = dec.receive_frame() {
                    criterion::black_box(frame);
                }
            }
        });
    });
    g.finish();
}

fn bench_decode_erased_5s(c: &mut Criterion) {
    // 5 s of erasures = 5 * 33 packets. Each is one byte (`0b11`).
    let erased: Vec<Packet> = (0..(5 * 33)).map(|_| packet(vec![0b11u8])).collect();
    let out_bytes = (erased.len() * FRAME_SAMPLES * 2) as u64;
    let mut g = c.benchmark_group("decode_erased_5s");
    g.throughput(Throughput::Bytes(out_bytes));
    g.bench_function(BenchmarkId::from_parameter("erased/8k/5s"), |b| {
        b.iter(|| {
            let mut dec = make_decoder();
            for pkt in &erased {
                dec.send_packet(pkt).expect("send_packet");
                while let Ok(frame) = dec.receive_frame() {
                    criterion::black_box(frame);
                }
            }
        });
    });
    g.finish();
}

fn bench_decode_mixed_5s(c: &mut Criterion) {
    // 5 s of mixed-rate traffic: alternating MP-MLQ, ACELP, erased,
    // and SID packets. Exercises the rate-discriminator dispatch
    // branch + the state transitions between the three frame types.
    let mpmlq = encoded_voiced_packets(2, Some(6_300));
    let acelp = encoded_voiced_packets(2, Some(5_300));
    let mut mixed: Vec<Packet> = Vec::with_capacity(5 * 33);
    let mut i = 0;
    while mixed.len() < 5 * 33 {
        match i % 4 {
            0 => mixed.push(mpmlq[i % mpmlq.len()].clone()),
            1 => mixed.push(acelp[i % acelp.len()].clone()),
            2 => mixed.push(packet(vec![0b11u8])),
            _ => {
                let mut sid = vec![0u8; 4];
                sid[0] = 0b10;
                mixed.push(packet(sid));
            }
        }
        i += 1;
    }
    let out_bytes = (mixed.len() * FRAME_SAMPLES * 2) as u64;
    let mut g = c.benchmark_group("decode_mixed_5s");
    g.throughput(Throughput::Bytes(out_bytes));
    g.bench_function(BenchmarkId::from_parameter("mixed/8k/5s"), |b| {
        b.iter(|| {
            let mut dec = make_decoder();
            for pkt in &mixed {
                dec.send_packet(pkt).expect("send_packet");
                while let Ok(frame) = dec.receive_frame() {
                    criterion::black_box(frame);
                }
            }
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_decode_mpmlq_synth_1s,
    bench_decode_acelp_synth_1s,
    bench_decode_erased_5s,
    bench_decode_mixed_5s,
);
criterion_main!(benches);
