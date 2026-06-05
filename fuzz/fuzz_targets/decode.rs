#![no_main]

//! Decode arbitrary fuzz-supplied packet streams through the
//! registered G.723.1 `Decoder` trait surface.
//!
//! G.723.1 is a stream codec: a real decoder sees a sequence of
//! 0/1/4/20/24-byte packets back-to-back, each whose 2-bit rate
//! discriminator selects one of the four `SynthesisState` entry
//! points (MP-MLQ, ACELP, SID, untransmitted/erasure). State carries
//! across packets — previous-frame LSP, the adaptive-codebook
//! excitation history, the LPC synthesis filter memory, the pitch
//! and formant post-filter taps, the tilt compensator's last sample,
//! the smoothed-AGC integrator — so the contract under fuzz isn't
//! just "each call must not panic in isolation": it's "after any
//! attacker-chosen sequence of `send_packet` calls, the decoder must
//! still be in a state where the next `send_packet` returns rather
//! than aborting." Single-packet harnesses miss the cross-rate
//! state-transition shapes that the synthesis pipeline actually has
//! to survive.
//!
//! Per-input shape: the fuzz blob is sliced into packets by reading a
//! 1-byte length prefix (`min(len, remaining)`) and consuming that
//! many bytes; the loop continues until the blob is exhausted, with
//! a hard cap of 256 packets per input to keep each libFuzzer
//! iteration sub-millisecond on the steady-state path. Empty slices
//! are passed through to the empty-packet rejection branch in
//! `Decoder::send_packet`. After each `send_packet`, `receive_frame`
//! is drained to completion so the `pending: VecDeque<Frame>`
//! pop path is exercised on every input — a regression there would
//! otherwise hide behind the `send_packet`-only contract.
//!
//! Reference: ITU-T G.723.1 (May 2006) §5.4 / Annex B Table B.1.

use libfuzzer_sys::fuzz_target;
use oxideav_core::packet::PacketFlags;
use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, Packet, Rational, SampleFormat, TimeBase,
};
use oxideav_g7231::CODEC_ID_STR;

/// G.723.1 is fixed at 8 kHz / mono / S16, mirroring the encoder's
/// validated `CodecParameters` shape.
const SAMPLE_RATE: u32 = 8_000;
/// Cap on packets per fuzz input. The steady-state `receive_frame`
/// drain after each `send_packet` keeps `pending` empty so this is
/// purely an iteration-time bound, not a memory bound.
const MAX_PACKETS_PER_INPUT: usize = 256;

fn make_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
    p.sample_rate = Some(SAMPLE_RATE);
    p.channels = Some(1);
    p.sample_format = Some(SampleFormat::S16);
    p
}

fn make_packet(data: Vec<u8>) -> Packet {
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

fuzz_target!(|data: &[u8]| {
    let mut reg = CodecRegistry::new();
    oxideav_g7231::register_codecs(&mut reg);
    let params = make_params();
    let mut dec = match reg.first_decoder(&params) {
        Ok(d) => d,
        // No registered decoder is a structural test failure that
        // the unit tests already pin, not a fuzz finding. Silently
        // return rather than aborting the libFuzzer loop.
        Err(_) => return,
    };

    let mut cur = data;
    let mut packets_seen = 0usize;
    while !cur.is_empty() && packets_seen < MAX_PACKETS_PER_INPUT {
        // 1-byte length prefix, clamped to the remaining slice. A
        // length of 0 produces an empty packet — the empty-packet
        // rejection branch in `Decoder::send_packet` is part of the
        // contract under test.
        let want = cur[0] as usize;
        cur = &cur[1..];
        let take = want.min(cur.len());
        let payload = cur[..take].to_vec();
        cur = &cur[take..];

        let pkt = make_packet(payload);
        // `send_packet` may legitimately reject (`Error::invalid`)
        // on malformed framing; that's the typed-error contract.
        // What we forbid is a panic, an integer overflow in a debug
        // build, an out-of-bounds index, or an attacker-controlled
        // allocation, all of which would crash libFuzzer here.
        let _ = dec.send_packet(&pkt);

        // Drain `pending` so the `receive_frame` `pop_front` path is
        // exercised on every input; the discarded `Frame` is
        // intentionally black-boxed via the `let _` binding.
        while let Ok(frame) = dec.receive_frame() {
            let _ = frame;
        }

        packets_seen += 1;
    }

    // `reset()` after the stream so the per-input reset path is
    // covered too — a panic there would never surface in a
    // single-shot harness.
    let _ = dec.reset();
});
