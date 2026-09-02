#![no_main]

//! Annex A silence-compression fuzzer: attacker-supplied PCM through
//! `SpecEncoder` with the **VAD on** and a fuzzer-driven per-frame rate
//! schedule, every emitted packet (24 / 20 / 4 / 1 octets) fed back
//! through the fixed-point decoder.
//!
//! Round 455. The `roundtrip` target drives the registry encoder, which
//! never enables Annex A; this target closes that gap. The annex path
//! is full of corners real speech never reaches: the A.2 inverse
//! filter with a noise filter taken from a degenerate past-average
//! autocorrelation, Durbin on all-zero / single-spike autocorrelations
//! (A.4.2), the Itakura distance on tiny energies, the pseudo-log SID
//! gain quantiser at its bounds, the A.4.5 quadratic gain fit with a
//! non-positive discriminant, and the SID / untransmitted octets
//! landing in a decoder that treats them as erasures. The contract is
//! panic-freedom (no debug overflow, no out-of-bounds index, no NaN
//! escaping into an index) across the whole loop.
//!
//! ## Fuzz input layout
//!
//! ```text
//!   byte 0      — control:
//!                   bit 0 → initial rate (0 = 6.3 kbit/s, 1 = 5.3 kbit/s)
//!                   bit 1 → high-pass filter off
//!                   bit 2 → toggle the rate every frame (§1.2 rate
//!                           switching at a frame boundary)
//!                   bit 3 → decode with the post-filter off
//!   bytes 1..   — little-endian i16 PCM, bounded to 64 frames.
//! ```

use libfuzzer_sys::fuzz_target;
use oxideav_g7231::encoder::SpecEncoder;
use oxideav_g7231::linepack::PackedRate;
use oxideav_g7231::qdec::QSynthesis;

const FRAME_SAMPLES: usize = 240;
const MAX_FRAMES: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let control = data[0];
    let mut rate = if control & 0b0001 != 0 {
        PackedRate::Low
    } else {
        PackedRate::High
    };
    let hp_off = control & 0b0010 != 0;
    let toggle = control & 0b0100 != 0;
    let pf_off = control & 0b1000 != 0;

    let pcm: Vec<i16> = data[1..]
        .chunks_exact(2)
        .take(MAX_FRAMES * FRAME_SAMPLES)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let mut enc = SpecEncoder::new(rate);
    enc.set_highpass(!hp_off);
    enc.set_vad(true);
    let mut dec = QSynthesis::new();
    dec.set_postfilter(!pf_off);

    for chunk in pcm.chunks(FRAME_SAMPLES) {
        let mut frame = [0i16; FRAME_SAMPLES];
        frame[..chunk.len()].copy_from_slice(chunk);
        enc.set_rate(rate);
        let bytes = enc.encode_frame(&frame);
        match bytes[0] & 0b11 {
            0b00 => {
                assert_eq!(bytes.len(), 24);
                let _ = dec.decode_mpmlq(&bytes).expect("self-produced high-rate frame");
            }
            0b01 => {
                assert_eq!(bytes.len(), 20);
                let _ = dec.decode_acelp(&bytes).expect("self-produced low-rate frame");
            }
            0b10 => {
                assert_eq!(bytes.len(), 4);
                assert!(oxideav_g7231::annex_a::unpack_sid(&bytes).is_some());
                let _ = dec.decode_erased();
            }
            _ => {
                assert_eq!(bytes.len(), 1);
                let _ = dec.decode_erased();
            }
        }
        if toggle {
            rate = match rate {
                PackedRate::High => PackedRate::Low,
                PackedRate::Low => PackedRate::High,
            };
        }
    }
});
