#![no_main]

//! Structured adversarial fuzzer for the G.723.1 **bitstream parser**
//! and per-rate frame decoders, below the `Decoder` trait surface.
//!
//! Round 286 FUZZ-depth lane. The round-236 `decode` target drives the
//! public `Decoder` trait with free-form bytes; this target instead
//! *constructs* frames that are structurally close to legal — correct
//! length, correct rate discriminator — then surgically corrupts a
//! single field (LSP split index, absolute/delta lag, gain index, FCB
//! pulse word, MP-MLQ reserved tail) or truncates the payload at an
//! exact field boundary. The aim is to reach the deep parse/dequant
//! paths inside `decode_acelp` / `decode_mpmlq` that free-form bytes
//! reach only by luck, and to stress the `BitReader::read_u32`
//! out-of-bits guard at every boundary.
//!
//! ## Surfaces driven on every iteration
//!
//! 1. `header::parse_frame_type` on the attacker's first byte (rate
//!    discriminator decode, §3.7).
//! 2. `bitreader::BitReader` — a direct, field-shaped read schedule
//!    (2 + 8·3 + 7 + 2 + 7 + 2 + 12·4 + 1·4 + …) so the LSB-first
//!    cross-byte assembly and the `out-of-bits` rejection are both hit
//!    on truncated buffers.
//! 3. `encoder::decode_acelp_local` and `encoder::decode_mpmlq_local`
//!    — the stateless per-rate decoders — on a payload whose rate byte
//!    matches the requested rate but whose body fields are corrupt.
//! 4. The stateful `SynthesisState::decode_*` via a short
//!    `Decoder::send_packet` sequence of these crafted frames, so
//!    field corruption is also seen by the cross-frame postfilter /
//!    erasure state, not just a cold-start decoder.
//!
//! ## Why field-targeted corruption
//!
//! A free-form byte fuzzer that happens to set the rate byte to `00`
//! still spends most of its energy on the length check and the rate
//! mismatch branch. Field-targeted corruption guarantees the bytes
//! *past* the discriminator are reached: a deliberately out-of-range
//! LSP split index exercises `dequantise_lsp`'s table bound; a maxed
//! delta-lag index exercises `decode_delta_lag`'s clamp; a saturated
//! gain word exercises the joint-gain dequant; an all-ones FCB word
//! exercises `unpack_fcb_bits` / `unpack_mpmlq_pulses` pulse placement
//! (where a bad position could index a 60-sample subframe out of
//! bounds if unclamped). All must return a frame or an `Err`, never
//! panic.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0      — control:
//!                   bit 0    → base rate (0 = MP-MLQ 24 B, 1 = ACELP 20 B)
//!                   bits 1-3 → truncation mode (0 = full frame, else
//!                              cut the payload to a field boundary,
//!                              probing the BitReader out-of-bits guard)
//!                   bits 4-7 → number of crafted frames to chain (1..16)
//!   byte 1      — corruption selector (which field to overwrite)
//!   bytes 2..   — raw bytes copied verbatim into the frame body after
//!                 the rate byte; the corruption selector then stamps a
//!                 chosen field with an extreme value.
//! ```

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Decoder, Packet, Rational, TimeBase};
use oxideav_g7231::bitreader::BitReader;
use oxideav_g7231::encoder::{decode_acelp_local, decode_mpmlq_local};
use oxideav_g7231::header::{parse_frame_type, FrameType};

const ACELP_BYTES: usize = 20;
const MPMLQ_BYTES: usize = 24;
const SAMPLE_RATE: u32 = 8_000;

/// Field-boundary bit offsets shared by both rates (the common prefix:
/// rate + 3 LSP splits + 4 lag indices + 4 gains + 4 grid bits). Used
/// to truncate a frame exactly at a field edge so the `BitReader`
/// out-of-bits guard is exercised on a sub-byte remainder, not just a
/// whole-byte short read.
const TRUNCATION_BIT_OFFSETS: &[u32] = &[
    2,  // after rate
    10, // after LSP split 0
    18, // after LSP split 1
    26, // after LSP split 2
    33, // after abs lag 0
    35, // after delta lag 1
    42, // after abs lag 2
    44, // after delta lag 3
    56, // after gain 0
    92, // after gain 3
    96, // after grid bits
];

/// Run a direct field-shaped BitReader schedule so the LSB-first
/// cross-byte assembly and the out-of-bits guard are both exercised
/// regardless of how the per-rate decoder happens to read.
fn drive_bitreader(body: &[u8]) {
    let mut br = BitReader::new(body);
    // Mirror the field widths the real decoders use; ignore values.
    let _ = br.read_u32(2); // rate
    for _ in 0..3 {
        let _ = br.read_u32(8); // LSP splits
    }
    let _ = br.read_u32(7); // abs lag 0
    let _ = br.read_u32(2); // delta lag 1
    let _ = br.read_u32(7); // abs lag 2
    let _ = br.read_u32(2); // delta lag 3
    for _ in 0..4 {
        let _ = br.read_u32(12); // gains
    }
    for _ in 0..4 {
        let _ = br.read_bit(); // grid
    }
    // Drain whatever remains a bit at a time until the guard fires.
    while br.read_bit().is_ok() {}
    // bit_position / bits_remaining must stay consistent after the
    // guard trips.
    let _ = br.bit_position();
    let _ = br.bits_remaining();
    // A zero-width read is always Ok.
    let _ = br.read_u32(0);
}

/// Stamp an extreme value into one field of a frame body to force the
/// deep dequant/placement paths. `selector` picks the field; `body`
/// already has its rate byte set.
fn corrupt_field(body: &mut [u8], selector: u8) {
    let len = body.len();
    if len < 4 {
        return;
    }
    match selector % 8 {
        0 => body[1] = 0xFF, // LSP split 0 — out-of-table-range index
        1 => body[2] = 0xFF, // LSP split 1
        2 => body[3] = 0xFF, // LSP split 2
        3 => {
            // Saturate the lag bytes region.
            if len > 4 {
                body[4] = 0xFF;
            }
        }
        4 => {
            // Saturate the gain region.
            for b in body.iter_mut().skip(5).take(6) {
                *b = 0xFF;
            }
        }
        5 => {
            // All-ones FCB / pulse region (tail of the frame).
            let start = len.saturating_sub(8);
            for b in body.iter_mut().skip(start) {
                *b = 0xFF;
            }
        }
        6 => {
            // Alternating bit pattern across the whole body (after rate).
            for (i, b) in body.iter_mut().enumerate().skip(1) {
                *b = if i % 2 == 0 { 0xAA } else { 0x55 };
            }
        }
        _ => {
            // Last byte (MP-MLQ reserved tail) maxed.
            body[len - 1] = 0xFF;
        }
    }
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
    if data.len() < 2 {
        // Still exercise the header parser on a 0/1-byte buffer.
        let _ = parse_frame_type(data);
        return;
    }
    let control = data[0];
    let corrupt_sel = data[1];
    let low_rate = control & 0b0001 != 0;
    let trunc_mode = (control >> 1) & 0b0111;
    let frame_count = ((control >> 4) as usize % 15) + 1;
    let payload_bytes = if low_rate { ACELP_BYTES } else { MPMLQ_BYTES };
    let rate_byte = if low_rate { 0b01u8 } else { 0b00u8 };

    let raw = &data[2..];

    // Always exercise the standalone header parser on the raw tail too.
    let _ = parse_frame_type(raw);

    let tb = TimeBase(Rational::new(1, SAMPLE_RATE as i64));
    let mut dec = build_decoder();

    // Build `frame_count` crafted frames, advancing through `raw` so
    // each frame draws fresh body bytes; wrap when we run out.
    for f in 0..frame_count {
        // Assemble a full-length body, rate byte forced, rest from raw.
        let mut body = vec![0u8; payload_bytes];
        body[0] = rate_byte;
        let base = (f * 7) % raw.len().max(1);
        for (i, slot) in body.iter_mut().enumerate().skip(1) {
            if let Some(&b) = raw.get((base + i) % raw.len().max(1)) {
                *slot = b;
            }
        }
        // Re-stamp the rate byte (the copy above may have clobbered it)
        // so the rate-match branch in decode_* is reached, then corrupt
        // a chosen field.
        body[0] = rate_byte;
        corrupt_field(&mut body, corrupt_sel.wrapping_add(f as u8));

        // Direct BitReader stress on the full body.
        drive_bitreader(&body);

        // Optionally truncate to a field boundary to probe the
        // out-of-bits guard on a sub-byte remainder.
        let frame_bytes: Vec<u8> = if trunc_mode == 0 {
            body.clone()
        } else {
            let off = TRUNCATION_BIT_OFFSETS
                [(trunc_mode as usize - 1).min(TRUNCATION_BIT_OFFSETS.len() - 1)];
            // Round up to whole bytes; this still leaves the decoder a
            // short buffer (its own length check rejects before the
            // BitReader on most paths — both branches are valid).
            let keep = (off as usize).div_ceil(8);
            body[..keep.min(body.len())].to_vec()
        };

        // Stateless per-rate decoder (cold-start each call).
        if low_rate {
            let _ = decode_acelp_local(&frame_bytes);
        } else {
            let _ = decode_mpmlq_local(&frame_bytes);
        }

        // Stateful path: feed through the trait decoder so the crafted
        // (possibly corrupt/truncated) frame is also seen by the
        // cross-frame postfilter + erasure state.
        if let Some(d) = dec.as_mut() {
            let pkt = Packet::new(0, tb, frame_bytes.clone());
            if d.send_packet(&pkt).is_ok() {
                let _ = d.receive_frame();
            }
        }
    }

    // Drain.
    if let Some(d) = dec.as_mut() {
        let _ = d.flush();
        for _ in 0..4 {
            if d.receive_frame().is_err() {
                break;
            }
        }
    }

    // Independently confirm the four discriminator branches all parse
    // (a regression guard cheap enough to run every iteration).
    for b in [0b00u8, 0b01, 0b10, 0b11] {
        if let Ok(ft) = parse_frame_type(&[b]) {
            let _ = ft.frame_size();
            let _ = ft.bit_rate_label();
            debug_assert_eq!(ft, FrameType::from_bits(b));
        }
    }
});
