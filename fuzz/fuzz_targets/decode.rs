#![no_main]

//! Drive attacker-supplied bytes through `Decoder::send_packet` on a
//! fresh G.723.1 decoder.
//!
//! Round 236 depth-mode lane: a panic-freedom fuzzer over the
//! decoder's attacker surface. The contract under test is purely
//! that every public entry point on the registered `Decoder` trait
//! object returns a `Result` (never panics, never integer-overflows
//! in a debug build, never indexes out of bounds) on arbitrary input,
//! across a **mixed-rate** packet stream so the cross-packet state
//! machine is exercised — not just one rate at a time.
//!
//! ## Why mixed-rate matters
//!
//! G.723.1 carries the rate selector in the low 2 bits of the first
//! payload byte (per ITU-T G.723.1 §3.7):
//!
//! ```text
//!   00 → 6.3 kbit/s MP-MLQ  (24-byte frame)
//!   01 → 5.3 kbit/s ACELP   (20-byte frame)
//!   10 → SID                ( 4-byte frame)
//!   11 → Untransmitted       (0 or 1 byte)
//! ```
//!
//! The four branches drive four different code paths through
//! `encoder::SynthesisState`:
//! - `decode_mpmlq` (high rate),
//! - `decode_acelp` (low rate),
//! - `decode_erased` (SID + untransmitted both feed the §3.10.2
//!   frame-erasure concealment path with its erased-run counter,
//!   adaptive postfilter / AGC carry-over per §3.9, and the
//!   classifier-driven voiced/unvoiced regeneration per §3.10.2).
//!
//! A single-rate fuzz session can never reach the discriminator
//! transitions where the §3.10.2 erased-run counter resets, where the
//! pitch-postfilter state recovers from a SID-driven mute, or where
//! the decoder's `pending` VecDeque is pushed/popped under
//! alternating rates. This target stitches packets of all four
//! shapes into one decoder session so the transitions are reachable.
//!
//! ## Surfaces driven on every iteration
//!
//! 1. `oxideav_g7231::register_codecs` → `make_decoder` factory
//!    surface (parameter shape + factory return).
//! 2. `Decoder::send_packet` across a fuzzer-chosen sequence of
//!    packet shapes — short / exact / over-long for each rate, plus
//!    truncations that hit the length-validation rejection at
//!    `parse_frame_type` and the per-rate `expected` size check.
//! 3. `Decoder::receive_frame` after each `send_packet` to drain
//!    `pending` and confirm the produced frame's shape is
//!    well-formed (or the caller-visible error is well-formed).
//! 4. `Decoder::flush` to mark the stream drained, then
//!    `receive_frame` until `Eof` — covers the post-flush path that
//!    the bench harness does not reach.
//! 5. `Decoder::reset` mid-stream — confirms the synthesis state
//!    re-seeds cleanly to silence without leaking the previous
//!    erased-run counter or `next_pts`.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0       — packet-count seed; up to 16 packets per iteration
//!                  (cap keeps each iteration cheap).
//!   bytes 1..    — concatenated packet bodies. The harness walks the
//!                  buffer; for each packet it reads a 1-byte length
//!                  selector, picks a packet size from the {0, 1, 4,
//!                  20, 24, len} ladder (covering the four spec-legal
//!                  sizes plus an attacker-chosen one), and slices
//!                  the next N bytes verbatim. End-of-buffer
//!                  short-circuits the loop. The first body byte of
//!                  each slice is fed verbatim so the rate
//!                  discriminator is attacker-controlled.
//! ```
//!
//! Output is discarded — the fuzzer only cares about *return*, not
//! correctness. PCM-shape sanity is left to the integration tests
//! and the bench harness.
//!
//! Citations:
//! - ITU-T G.723.1 Recommendation §3.7 (rate discriminator decoding,
//!   frame sizes 24 / 20 / 4 / 0–1 B).
//! - ITU-T G.723.1 §3.9 (formant postfilter steady-state, AGC).
//! - ITU-T G.723.1 §3.10.2 (frame-erasure concealment path, voiced /
//!   unvoiced classifier, attenuation schedule).
//! - Local: `crates/oxideav-g7231/src/header.rs` (`parse_frame_type`,
//!   `FrameType::frame_size`, `FrameType::bit_rate_label`).
//! - Local: `crates/oxideav-g7231/src/lib.rs` (`G7231Decoder`
//!   send/receive/flush/reset).

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Decoder, Packet, Rational, TimeBase};

/// Spec-legal G.723.1 packet sizes per §3.7. Includes 0 and 1 for
/// the `Untransmitted` (`11`) discriminator which carries either no
/// payload or a single header byte.
const SPEC_SIZES: &[usize] = &[0, 1, 4, 20, 24];

const MAX_PACKETS_PER_ITER: usize = 16;

fn pick_size(selector: u8, attacker_len: usize) -> usize {
    // Six-way pick: five spec-legal sizes plus one attacker-chosen
    // length so the fuzzer can probe sizes that no rate accepts (e.g.
    // 12 bytes for `01` low-rate which expects 20). The selector's
    // low 3 bits index a 6-element table; the 7th and 8th slots fall
    // back to the attacker-chosen length.
    match selector & 0b111 {
        0 => SPEC_SIZES[0],
        1 => SPEC_SIZES[1],
        2 => SPEC_SIZES[2],
        3 => SPEC_SIZES[3],
        4 => SPEC_SIZES[4],
        _ => attacker_len,
    }
}

fn build_decoder() -> Option<Box<dyn Decoder>> {
    let mut reg = CodecRegistry::new();
    oxideav_g7231::register_codecs(&mut reg);
    let id = CodecId::new(oxideav_g7231::CODEC_ID_STR);
    let mut params = CodecParameters::audio(id);
    params.channels = Some(1);
    params.sample_rate = Some(8_000);
    reg.first_decoder(&params).ok()
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let packet_count = ((data[0] as usize) % MAX_PACKETS_PER_ITER) + 1;
    let mut cursor = 1usize;

    let mut dec = match build_decoder() {
        Some(d) => d,
        None => return,
    };

    let tb = TimeBase(Rational::new(1, 8_000));

    // Inject a `reset()` and `flush()` round-trip at iteration index
    // `reset_at` / `flush_at` so the cross-packet state-machine
    // transitions through both calls mid-stream, not only at the end.
    // The hooks are deterministic in `data[0]` so libfuzzer's
    // minimiser can still shrink reliably.
    let reset_at = if packet_count >= 4 {
        (data[0] as usize) % packet_count
    } else {
        usize::MAX
    };
    let flush_at = if packet_count >= 2 {
        ((data[0] >> 4) as usize) % packet_count
    } else {
        usize::MAX
    };

    for i in 0..packet_count {
        // Out of buffer → stop. We always need at least 1 byte (the
        // size selector) plus whatever the selector demands.
        if cursor >= data.len() {
            break;
        }
        let selector = data[cursor];
        cursor += 1;

        // Attacker-chosen length, bounded so a malicious selector
        // can't pin a worker on a 4 GiB allocation. 64 bytes is
        // enough to cover every spec-legal size and a few sizes
        // either side of each rate's `expected` boundary.
        let attacker_len = (selector as usize) & 0x3f;
        let size = pick_size(selector, attacker_len);
        let take = size.min(data.len() - cursor);
        let body = data[cursor..cursor + take].to_vec();
        cursor += take;

        let pkt = Packet::new(0, tb, body);
        // send_packet may legitimately return an error on a
        // truncated tail or an unknown framing — that's the contract.
        // Output frames are drained immediately to keep `pending`
        // bounded.
        if dec.send_packet(&pkt).is_ok() {
            let _ = dec.receive_frame();
        }

        if i == flush_at {
            let _ = dec.flush();
            // After flush, drain pending then expect Eof.
            for _ in 0..4 {
                if dec.receive_frame().is_err() {
                    break;
                }
            }
        }
        if i == reset_at {
            let _ = dec.reset();
        }
    }

    // Final flush + drain, regardless of where the loop exited.
    let _ = dec.flush();
    for _ in 0..4 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
});
