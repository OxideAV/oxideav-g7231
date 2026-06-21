#![no_main]

//! Parameter-validation + erasure-concealment panic-freedom fuzzer for
//! the G.723.1 codec factories.
//!
//! The `decode`, `roundtrip`, and `bitstream` targets all build their
//! encoder/decoder from a *fixed*, spec-legal `CodecParameters`
//! (8 kHz / mono / S16 / 5300|6300). They therefore never exercise the
//! factory's own validation surface — the branches in
//! `encoder::make_encoder` that reject a wrong sample rate, a non-mono
//! channel count, a non-S16 sample format, or an out-of-range bit rate,
//! and the bit-rate→mode selection at the exact 5.3 / 6.3 kbit/s
//! acceptance-window boundaries (ITU-T G.723.1 §1, dual-rate operation).
//!
//! This target drives `make_encoder` / `make_decoder` with
//! attacker-chosen parameters drawn from a deliberately wide ladder
//! (including the acceptance-window edges 4999/5000/5600/5601 and
//! 5999/6000/6500/6501 around each rate's band), confirming the factory
//! always returns a well-formed `Result` — a usable codec or a clean
//! `Err`, never a panic.
//!
//! ## Sustained erasure / SID decay
//!
//! When a decoder *is* successfully built, the second half of the input
//! drives a long run of SID (`10`) and untransmitted (`11`) packets
//! interleaved with the occasional valid frame and a `reset()`, so the
//! §3.10.2 erasure-concealment state machine — the per-frame 2.5 dB
//! attenuation schedule, the mute-after-N-frames latch, the §3.10.1 LSP
//! leak toward the DC vector, and the pseudo-random innovation seed — is
//! walked through its full decay-and-recovery cycle and then re-seeded
//! by `reset()`. The other three targets reach erasure frames only
//! incidentally; here the erased run is the focus, so the deep decay
//! arithmetic (the `powf`, the LSP angular-frequency conversion) is
//! exercised at every attenuation step including the post-mute floor.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0      — sample-rate selector (indexes a small ladder incl.
//!                 8000 and several rejected rates)
//!   byte 1      — channel-count selector (1 accepted; others rejected)
//!   byte 2      — sample-format selector (S16 accepted; others rejected)
//!   byte 3      — bit-rate selector (band edges + out-of-band values)
//!   byte 4      — erasure-sequence length / shape seed
//!   bytes 5..   — per-step opcodes for the decoder drive loop (each low
//!                 2 bits pick SID / untransmitted / a valid frame /
//!                 reset); body bytes for valid frames are taken
//!                 verbatim from the tail.
//! ```
//!
//! Output is discarded — only *return* is asserted.
//!
//! Citations:
//! - ITU-T G.723.1 Recommendation §1 (dual-rate 5.3 / 6.3 kbit/s).
//! - ITU-T G.723.1 §3.10.1 / §3.10.2 (erasure concealment: LSP leak,
//!   attenuation schedule, mute latch, innovation seed).
//! - Local: `crates/oxideav-g7231/src/encoder.rs` (`make_encoder`
//!   parameter validation, bit-rate→mode bands; `decode_erased`).
//! - Local: `crates/oxideav-g7231/src/lib.rs` (`make_decoder`,
//!   `G7231Decoder` send/receive/flush/reset).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Decoder, Packet, Rational, SampleFormat, TimeBase,
};

const SAMPLE_RATE: u32 = 8_000;

/// Sample-rate ladder: the one accepted rate plus a spread of rejected
/// ones (including 0, which `make_encoder` reads via `unwrap_or` only
/// when the field is `None` — here it is `Some`, so 0 must be rejected).
const SR_LADDER: &[u32] = &[8_000, 0, 1, 7_999, 8_001, 16_000, 44_100, 48_000, u32::MAX];

/// Channel ladder: 1 accepted; the rest rejected.
const CH_LADDER: &[u16] = &[1, 0, 2, 6, 255, u16::MAX];

/// Sample-format ladder: S16 accepted; the rest rejected.
const FMT_LADDER: &[SampleFormat] = &[
    SampleFormat::S16,
    SampleFormat::U8,
    SampleFormat::S8,
    SampleFormat::S24,
    SampleFormat::S32,
    SampleFormat::F32,
    SampleFormat::F64,
    SampleFormat::S16P,
];

/// Bit-rate ladder: both accepted bands' interiors and edges, plus
/// rejected values straddling each acceptance window
/// (`6000..=6500` → MP-MLQ, `5000..=5600` → ACELP per `make_encoder`).
const BR_LADDER: &[Option<u64>] = &[
    None,
    Some(6_300),
    Some(5_300),
    Some(6_000),
    Some(6_500),
    Some(5_000),
    Some(5_600),
    Some(5_999), // just below MP-MLQ band, above ACELP band → reject
    Some(6_501), // just above MP-MLQ band → reject
    Some(4_999), // just below ACELP band → reject
    Some(5_601), // just above ACELP band, below MP-MLQ band → reject
    Some(0),
    Some(u64::MAX),
];

fn pick<T: Copy>(table: &[T], sel: u8) -> T {
    table[sel as usize % table.len()]
}

fn build_params(sr: u32, ch: u16, fmt: SampleFormat, br: Option<u64>) -> CodecParameters {
    let id = CodecId::new(oxideav_g7231::CODEC_ID_STR);
    let mut p = CodecParameters::audio(id);
    p.sample_rate = Some(sr);
    p.channels = Some(ch);
    p.sample_format = Some(fmt);
    p.bit_rate = br;
    p
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        // Still exercise the factory surface on a degenerate input: a
        // default-ish param set must produce a clean Result.
        let p = build_params(SAMPLE_RATE, 1, SampleFormat::S16, Some(6_300));
        let _ = oxideav_g7231::encoder::make_encoder(&p);
        let mut reg = CodecRegistry::new();
        oxideav_g7231::register_codecs(&mut reg);
        let _ = reg.first_decoder(&p);
        return;
    }

    let sr = pick(SR_LADDER, data[0]);
    let ch = pick(CH_LADDER, data[1]);
    let fmt = pick(FMT_LADDER, data[2]);
    let br = pick(BR_LADDER, data[3]);
    let params = build_params(sr, ch, fmt, br);

    // ---- factory validation surface ----
    // make_encoder must return Ok or a clean Err for every ladder combo.
    let enc_ok = oxideav_g7231::encoder::make_encoder(&params);
    if let Ok(mut enc) = enc_ok {
        // A built encoder's output params + a flush must not panic even
        // when no frame was ever sent.
        let _ = enc.output_params();
        let _ = enc.flush();
        while enc.receive_packet().is_ok() {}
    }

    // make_decoder via the registry. The decoder factory accepts any
    // params (it ignores rate/channel hints and always synthesises
    // 8 kHz mono), so this should generally build; drive it below.
    let mut reg = CodecRegistry::new();
    oxideav_g7231::register_codecs(&mut reg);
    let mut dec: Option<Box<dyn Decoder>> = reg.first_decoder(&params).ok();

    // ---- sustained erasure / SID decay drive ----
    let tb = TimeBase(Rational::new(1, SAMPLE_RATE as i64));
    let opcodes = &data[5..];
    // Bound the number of steps so a worker can't be pinned; the erasure
    // decay fully resolves (mute) within a handful of frames anyway.
    let steps = opcodes.len().min(96);

    // A spec-legal high-rate body (rate byte 00, 24 bytes) reused for the
    // "valid frame" opcode; its tail bytes are stamped from the input so
    // successive valid frames differ.
    let mut valid_hi = [0u8; 24];

    for (i, &op) in opcodes.iter().take(steps).enumerate() {
        let Some(d) = dec.as_mut() else { break };
        match op & 0b11 {
            0 => {
                // SID frame: rate byte 10, 4 bytes. send_packet routes it
                // through the erasure/concealment path after the length
                // check accepts the 4-byte payload.
                let body = vec![0b10u8, 0, 0, 0];
                let pkt = Packet::new(0, tb, body);
                if d.send_packet(&pkt).is_ok() {
                    let _ = d.receive_frame();
                }
            }
            1 => {
                // Untransmitted: rate byte 11, single byte → concealment.
                let pkt = Packet::new(0, tb, vec![0b11u8]);
                if d.send_packet(&pkt).is_ok() {
                    let _ = d.receive_frame();
                }
            }
            2 => {
                // Valid high-rate frame to break the erased run so the
                // decay→recovery transition is reached. Stamp the tail
                // from the input for variety; keep the rate byte at 00.
                valid_hi[0] = 0;
                for (k, slot) in valid_hi.iter_mut().enumerate().skip(1) {
                    *slot = op.wrapping_add((i + k) as u8);
                }
                let pkt = Packet::new(0, tb, valid_hi.to_vec());
                if d.send_packet(&pkt).is_ok() {
                    let _ = d.receive_frame();
                }
            }
            _ => {
                // Reset mid-decay: the erased-run counter, LSP leak, and
                // pseudo-random seed must re-initialise without leaking
                // state into the next frame.
                let _ = d.reset();
            }
        }
    }

    if let Some(mut d) = dec {
        let _ = d.flush();
        for _ in 0..4 {
            if d.receive_frame().is_err() {
                break;
            }
        }
    }
});
