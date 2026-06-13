#![no_main]

//! Drive attacker-supplied PCM through the G.723.1 **encoder** at a
//! fuzzer-chosen rate, then feed every packet the encoder emits back
//! into a fresh decoder — a closed-loop encode→decode panic-freedom
//! fuzzer.
//!
//! Round 286 FUZZ-depth lane. The round-203 bench harness exercises
//! the encoder only on a fixed sine sweep; the round-236 `decode`
//! target drives the decoder's *adversarial* surface but never the
//! encoder's. This target closes the gap: it pushes arbitrary 16-bit
//! PCM (sign-extended garbage, clipped extremes, silence, ramps) into
//! `Encoder::send_frame`, drains `receive_packet`, then routes those
//! self-produced bitstreams through the registered `Decoder`. The
//! contract under test is panic-freedom across the *whole* analysis →
//! bit-pack → parse → synthesis loop, on input the analysis path was
//! never tuned for.
//!
//! ## Why drive the encoder with garbage PCM
//!
//! The encoder's analysis path (autocorrelation → Levinson-Durbin →
//! Chebyshev LSP root-finding → closed-loop pitch + FCB search →
//! joint-gain quant) is full of divisions, `sqrt`/`log2` calls, and
//! fixed-size index math (ITU-T G.723.1 §2.x). Real speech keeps the
//! intermediate magnitudes well-conditioned; adversarial PCM does not.
//! A degenerate autocorrelation (e.g. an all-`i16::MIN` block, or a
//! ±full-scale square wave) can push Levinson's reflection-coefficient
//! recursion or the LSP root bracket into the corners the bench never
//! reaches. Anything the encoder emits must then *also* survive the
//! decoder, since the two share one `SynthesisState`.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0      — control:
//!                   bit 0  → rate select (0 = 6.3 kbit/s MP-MLQ,
//!                                         1 = 5.3 kbit/s ACELP)
//!                   bit 1  → inject a mid-stream `flush()` so the
//!                            tail-padding path (partial final frame,
//!                            §2.2 frame assembly) is reached
//!                   bit 2  → feed the encoder's packets to the
//!                            decoder in *reverse* arrival order so the
//!                            decoder's cross-packet state machine sees
//!                            a non-causal sequence
//!   bytes 1..   — little-endian i16 PCM samples. The encoder buffers
//!                 them into 240-sample (30 ms) frames; a trailing
//!                 partial frame is flushed (zero-padded) at `flush()`.
//! ```
//!
//! Output PCM is discarded — only *return* is asserted, never
//! sample-correctness (that is the integration tests' job).
//!
//! Citations:
//! - ITU-T G.723.1 Recommendation §2.x (encoder analysis: LPC, LSP,
//!   pitch + FCB search, gain quant, §2.2 frame assembly).
//! - ITU-T G.723.1 §3.7 (rate discriminator, frame sizes).
//! - Local: `crates/oxideav-g7231/src/encoder.rs` (`make_encoder`,
//!   `G7231Encoder` send/receive/flush, `analyse_*` / `pack_*`).
//! - Local: `crates/oxideav-g7231/src/lib.rs` (`G7231Decoder`).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, CodecRegistry, Decoder, Encoder, Frame, Packet, Rational,
    TimeBase,
};

const SAMPLE_RATE: u32 = 8_000;
const FRAME_SAMPLES: usize = 240;
/// Cap the PCM fed per iteration so a worker can't be pinned encoding a
/// multi-megabyte block. 64 frames (≈1.9 s of audio) is more than
/// enough to walk the encoder's cross-frame state several times over.
const MAX_SAMPLES: usize = FRAME_SAMPLES * 64;

fn build_encoder(low_rate: bool) -> Option<Box<dyn Encoder>> {
    let id = CodecId::new(oxideav_g7231::CODEC_ID_STR);
    let mut params = CodecParameters::audio(id);
    params.channels = Some(1);
    params.sample_rate = Some(SAMPLE_RATE);
    params.bit_rate = Some(if low_rate { 5_300 } else { 6_300 });
    oxideav_g7231::encoder::make_encoder(&params).ok()
}

fn build_decoder() -> Option<Box<dyn Decoder>> {
    let mut reg = CodecRegistry::new();
    oxideav_g7231::register_codecs(&mut reg);
    let id = CodecId::new(oxideav_g7231::CODEC_ID_STR);
    let mut params = CodecParameters::audio(id);
    params.channels = Some(1);
    params.sample_rate = Some(SAMPLE_RATE);
    reg.first_decoder(&params).ok()
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let control = data[0];
    let low_rate = control & 0b001 != 0;
    let mid_flush = control & 0b010 != 0;
    let reverse = control & 0b100 != 0;

    let tb = TimeBase(Rational::new(1, SAMPLE_RATE as i64));

    let mut enc = match build_encoder(low_rate) {
        Some(e) => e,
        None => return,
    };

    // Decode the rest of the buffer as little-endian i16 PCM, bounded.
    let pcm_bytes = &data[1..];
    let n_samples = (pcm_bytes.len() / 2).min(MAX_SAMPLES);
    let mut pcm: Vec<i16> = Vec::with_capacity(n_samples);
    for chunk in pcm_bytes.chunks_exact(2).take(n_samples) {
        pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }

    // Collect every packet the encoder emits.
    let mut packets: Vec<Packet> = Vec::new();

    // Split the PCM into two halves so we can fire a mid-stream flush
    // between them; the encoder's `eof` latch means a second flush is a
    // no-op, exercising the idempotent-flush guard.
    let split = pcm.len() / 2;
    let (first, second) = pcm.split_at(split.min(pcm.len()));

    let push_pcm = |enc: &mut Box<dyn Encoder>, samples: &[i16], packets: &mut Vec<Packet>| {
        if samples.is_empty() {
            return;
        }
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let af = AudioFrame {
            samples: samples.len() as u32,
            pts: None,
            data: vec![bytes],
        };
        if enc.send_frame(&Frame::Audio(af)).is_ok() {
            while let Ok(pkt) = enc.receive_packet() {
                packets.push(pkt);
            }
        }
    };

    push_pcm(&mut enc, first, &mut packets);
    if mid_flush {
        let _ = enc.flush();
        while let Ok(pkt) = enc.receive_packet() {
            packets.push(pkt);
        }
    }
    push_pcm(&mut enc, second, &mut packets);

    // Final flush: drains the partial trailing frame (zero-padded) per
    // §2.2 frame assembly. Idempotent after a mid-stream flush.
    let _ = enc.flush();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }

    if reverse {
        packets.reverse();
    }

    // Feed the self-produced bitstreams back through the decoder.
    let mut dec = match build_decoder() {
        Some(d) => d,
        None => return,
    };
    for pkt in &packets {
        let dpkt = Packet::new(0, tb, pkt.data.clone());
        if dec.send_packet(&dpkt).is_ok() {
            let _ = dec.receive_frame();
        }
    }
    let _ = dec.flush();
    for _ in 0..4 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
});
