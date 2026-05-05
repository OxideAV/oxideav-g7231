//! G.723.1 encoder -> decoder round-trip at both dual rates.
//!
//! Two kinds of round-trip are exercised here:
//!
//!   1. Encoder -> `decode_{acelp,mpmlq}_local` — the encoder's sister
//!      reference decoder, which inverts the *same* simplified VQ tables
//!      the encoder uses. This confirms both bit-packing layouts are
//!      self-consistent and that the analysis pipeline produces
//!      non-trivial excitation for speech-like input.
//!
//!   2. Encoder -> the registered framework decoder. The shipped decoder
//!      is today a silence-emitting scaffold that validates framing and
//!      emits 30 ms of zeros per packet (see the crate docstring). This
//!      end-to-end path is still worth exercising: it pins down the
//!      framing contract (rate discriminator, payload lengths, PTS) that
//!      any future full decoder must satisfy.

use oxideav_core::packet::PacketFlags;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, Packet, Rational, Result, SampleFormat, TimeBase,
};
use oxideav_core::{CodecRegistry, Decoder, Encoder};
use oxideav_g7231::encoder::{decode_acelp_local, decode_mpmlq_local};
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

fn audio_frame(samples: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    // Stream-level shape (S16 / mono / 8 kHz / time_base) lives on the
    // stream's CodecParameters now; the encoder reads them off
    // make_params() at construction.
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

/// Voiced synthetic signal with a 180 Hz fundamental plus three harmonics.
/// Exercises the pitch analyser, fixed-codebook search, and LSP quantiser.
fn voiced(frames: usize) -> Vec<i16> {
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

fn collect_packets(enc: &mut dyn Encoder) -> Vec<Packet> {
    let mut v = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        v.push(p);
    }
    v
}

#[test]
fn acelp_encoder_emits_20_byte_frames_with_01_discriminator() {
    let mut enc = make_encoder(Some(5300));
    let pcm = voiced(4);
    enc.send_frame(&audio_frame(&pcm)).unwrap();
    enc.flush().unwrap();
    let packets = collect_packets(&mut *enc);
    assert_eq!(packets.len(), 4);
    for p in &packets {
        assert_eq!(p.data.len(), 20);
        assert_eq!(p.data[0] & 0b11, 0b01, "ACELP discriminator = 01");
    }
}

#[test]
fn mpmlq_encoder_emits_24_byte_frames_with_00_discriminator() {
    let mut enc = make_encoder(Some(6300));
    let pcm = voiced(4);
    enc.send_frame(&audio_frame(&pcm)).unwrap();
    enc.flush().unwrap();
    let packets = collect_packets(&mut *enc);
    assert_eq!(packets.len(), 4);
    for p in &packets {
        assert_eq!(p.data.len(), 24);
        assert_eq!(p.data[0] & 0b11, 0b00, "MP-MLQ discriminator = 00");
    }
}

#[test]
fn acelp_roundtrip_through_local_decoder_preserves_energy() {
    const FRAMES: usize = 8;
    let input = voiced(FRAMES);
    let mut enc = make_encoder(Some(5300));
    enc.send_frame(&audio_frame(&input)).unwrap();
    enc.flush().unwrap();

    let mut out = Vec::new();
    for p in collect_packets(&mut *enc) {
        out.extend_from_slice(&decode_acelp_local(&p.data).unwrap());
    }
    assert_eq!(out.len(), FRAMES * FRAME_SAMPLES);

    let energy: f64 = out.iter().map(|&s| (s as f64).powi(2)).sum();
    let input_energy: f64 = input.iter().map(|&s| (s as f64).powi(2)).sum();
    assert!(
        energy >= 0.01 * input_energy,
        "ACELP roundtrip energy {energy:.3e} too small vs input {input_energy:.3e}"
    );
}

#[test]
fn mpmlq_roundtrip_through_local_decoder_preserves_energy() {
    const FRAMES: usize = 8;
    let input = voiced(FRAMES);
    let mut enc = make_encoder(Some(6300));
    enc.send_frame(&audio_frame(&input)).unwrap();
    enc.flush().unwrap();

    let mut out = Vec::new();
    for p in collect_packets(&mut *enc) {
        out.extend_from_slice(&decode_mpmlq_local(&p.data).unwrap());
    }
    assert_eq!(out.len(), FRAMES * FRAME_SAMPLES);

    let energy: f64 = out.iter().map(|&s| (s as f64).powi(2)).sum();
    let input_energy: f64 = input.iter().map(|&s| (s as f64).powi(2)).sum();
    assert!(
        energy >= 0.01 * input_energy,
        "MP-MLQ roundtrip energy {energy:.3e} too small vs input {input_energy:.3e}"
    );
}

#[test]
fn framework_decoder_consumes_both_rates() -> Result<()> {
    for bit_rate in [5300u64, 6300] {
        let mut enc = make_encoder(Some(bit_rate));
        enc.send_frame(&audio_frame(&voiced(3)))?;
        enc.flush()?;
        let mut dec = make_decoder();
        for pkt in collect_packets(&mut *enc) {
            dec.send_packet(&pkt)?;
            let f = dec.receive_frame()?;
            match f {
                Frame::Audio(af) => {
                    assert_eq!(af.samples, FRAME_SAMPLES as u32);
                    // Stream-level shape (sample_rate / channels /
                    // format) used to live on the frame; with the
                    // slim it lives on the stream's
                    // `CodecParameters` and is set by the registry
                    // factory off `make_params()`. The per-frame
                    // contract here is just the sample count + byte
                    // payload.
                    assert_eq!(af.data.len(), 1);
                    assert_eq!(af.data[0].len(), FRAME_SAMPLES * 2);
                }
                _ => panic!("expected audio frame"),
            }
        }
    }
    Ok(())
}

#[test]
fn framework_decoder_accepts_sid_and_untransmitted() -> Result<()> {
    let mut dec = make_decoder();
    // SID (0b10) with 4-byte payload.
    dec.send_packet(&packet(vec![0b10, 0, 0, 0]))?;
    let f = dec.receive_frame()?;
    let Frame::Audio(af) = f else {
        panic!("expected audio frame");
    };
    assert_eq!(af.samples, FRAME_SAMPLES as u32);

    // Untransmitted (0b11) — discriminator byte alone is legal.
    dec.send_packet(&packet(vec![0b11]))?;
    let f = dec.receive_frame()?;
    assert!(matches!(f, Frame::Audio(_)));
    Ok(())
}

/// Frame erasure concealment: dropping one packet in the middle of a
/// voiced stream must produce a concealed frame that (a) has non-zero
/// energy (so the listener doesn't hear a momentary silence hole), and
/// (b) the following good packet must decode cleanly — overall PSNR
/// stays above 10 dB, well above "broken".
#[test]
fn erasure_in_middle_of_stream_is_concealed() {
    const FRAMES: usize = 12;
    let input = voiced(FRAMES);
    let mut enc = make_encoder(Some(6300));
    enc.send_frame(&audio_frame(&input)).unwrap();
    enc.flush().unwrap();

    let mut dec = make_decoder();
    let packets = collect_packets(&mut *enc);
    let mut decoded: Vec<i16> = Vec::with_capacity(FRAMES * FRAME_SAMPLES);
    let erased_idx = 6; // drop the 7th frame
    let mut concealed_energy = 0.0f64;
    for (i, pkt) in packets.iter().enumerate() {
        if i == erased_idx {
            // Substitute an untransmitted packet so the decoder runs
            // concealment.
            dec.send_packet(&packet(vec![0b11])).unwrap();
            let Frame::Audio(af) = dec.receive_frame().unwrap() else {
                panic!("expected audio frame");
            };
            for chunk in af.data[0].chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f64;
                concealed_energy += s * s;
                decoded.push(s as i16);
            }
            continue;
        }
        dec.send_packet(pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        for chunk in af.data[0].chunks_exact(2) {
            decoded.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
    }

    // The concealed frame should carry some signal energy (not zero
    // silence) for the first erased frame in a run; the attenuation
    // schedule applies from the first frame at 0.7× so energy should be
    // well above zero.
    assert!(
        concealed_energy > 0.0,
        "concealed frame is pure silence ({concealed_energy})"
    );

    // And the overall decoded signal must remain coherent with the input.
    let mut mse = 0.0f64;
    for i in 0..decoded.len() {
        let e = decoded[i] as f64 - input[i] as f64;
        mse += e * e;
    }
    mse /= decoded.len() as f64;
    let psnr = 10.0 * (32_767.0f64.powi(2) / mse.max(1e-10)).log10();
    assert!(
        psnr > 10.0,
        "PSNR after erasure {psnr:.2} dB, expected > 10"
    );
}

/// A long run of erasures must eventually decay to silence (the
/// concealment attenuation schedule mutes after ~5 frames). Verifies the
/// decoder doesn't emit a buzzing runaway when the network drops many
/// packets in a row.
#[test]
fn sustained_erasure_run_decays_to_silence() {
    let mut dec = make_decoder();
    // Prime the decoder with a single good MP-MLQ frame.
    let mut enc = make_encoder(Some(6300));
    enc.send_frame(&audio_frame(&voiced(1))).unwrap();
    enc.flush().unwrap();
    for pkt in collect_packets(&mut *enc) {
        dec.send_packet(&pkt).unwrap();
        let _ = dec.receive_frame().unwrap();
    }

    // Drop 10 frames in a row.
    let mut energies = Vec::new();
    for _ in 0..10 {
        dec.send_packet(&packet(vec![0b11])).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        let mut e = 0.0f64;
        for chunk in af.data[0].chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f64;
            e += s * s;
        }
        energies.push(e);
    }
    // The last erasures in the run must be effectively silent.
    let late = energies[5..].iter().sum::<f64>();
    assert!(late < 1e6, "late erasure energy {late} did not mute");
}

#[test]
fn framework_decoder_rejects_short_high_rate_frame() {
    let mut dec = make_decoder();
    // Rate discriminator says high-rate (0b00) but the payload is far
    // shorter than the advertised 24 bytes.
    assert!(dec.send_packet(&packet(vec![0u8; 5])).is_err());
}

#[test]
fn pts_rises_monotonically_for_both_rates() {
    for bit_rate in [5300u64, 6300] {
        let mut enc = make_encoder(Some(bit_rate));
        enc.send_frame(&audio_frame(&voiced(6))).unwrap();
        enc.flush().unwrap();
        let mut last = -1i64;
        for p in collect_packets(&mut *enc) {
            let pts = p.pts.expect("packet PTS must be set");
            assert!(pts > last, "PTS must rise monotonically at {bit_rate}");
            last = pts;
        }
    }
}

/// Full encode → framework-decoder round-trip, both rates, two seconds
/// of voiced synthetic input, PSNR asserted above 15 dB per rate.
///
/// Observed on a representative run (x86_64, release):
///
/// - 5.3 kbit/s ACELP:  PSNR = 19–20 dB
/// - 6.3 kbit/s MP-MLQ: PSNR = 22–23 dB
///
/// The 15 dB floor leaves room for small DSP rounding drift across
/// platforms while still catching any regression that would send the
/// reconstruction back to the pre-fix ~5 dB level.
#[test]
fn roundtrip_two_seconds_voiced_psnr_both_rates() {
    // 2 s @ 8 kHz = 16000 samples = 66.7 frames → use 66 frames of 240.
    const FRAMES: usize = 66;
    let input = voiced(FRAMES);

    // Per-rate PSNR floors: ACELP has 4 pulses per subframe (vs 5-6 for
    // MP-MLQ) so it reconstructs less faithfully — 16 dB leaves ~2 dB of
    // cross-platform rounding headroom above the measured ~18 dB floor.
    // MP-MLQ steadily sits above 21 dB; 19 dB here is a catch-all for
    // cross-platform float drift.
    for (bit_rate, floor_db, label) in [(5300u64, 16.0, "ACELP"), (6300u64, 19.0, "MP-MLQ")] {
        let mut enc = make_encoder(Some(bit_rate));
        enc.send_frame(&audio_frame(&input)).unwrap();
        enc.flush().unwrap();

        let mut dec = make_decoder();
        let mut decoded: Vec<i16> = Vec::with_capacity(FRAMES * FRAME_SAMPLES);
        for pkt in collect_packets(&mut *enc) {
            dec.send_packet(&pkt).unwrap();
            let Frame::Audio(af) = dec.receive_frame().unwrap() else {
                panic!("expected audio frame");
            };
            assert_eq!(af.samples, FRAME_SAMPLES as u32);
            for chunk in af.data[0].chunks_exact(2) {
                decoded.push(i16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }

        // Some encoders may emit an extra partial-frame flush, so only
        // compare the first FRAMES*240 samples.
        let n = FRAMES * FRAME_SAMPLES;
        assert!(decoded.len() >= n, "{label}: got {} samples", decoded.len());
        let decoded = &decoded[..n];

        // PSNR (vs i16 full-scale peak).
        let mut mse = 0.0f64;
        for i in 0..n {
            let e = decoded[i] as f64 - input[i] as f64;
            mse += e * e;
        }
        mse /= n as f64;
        let peak = 32_767.0f64;
        let psnr = 10.0 * (peak * peak / mse.max(1e-10)).log10();

        // Signal-energy SNR.
        let mut sig_e = 0.0f64;
        for &s in input.iter().take(n) {
            sig_e += (s as f64).powi(2);
        }
        sig_e /= n as f64;
        let snr = 10.0 * (sig_e / mse.max(1e-10)).log10();

        eprintln!(
            "roundtrip_two_seconds_voiced_psnr_both_rates: {label} PSNR = {psnr:.2} dB, SNR = {snr:.2} dB"
        );
        assert!(
            psnr >= floor_db,
            "{label} PSNR = {psnr:.2} dB below floor {floor_db} dB"
        );
    }
}

/// Emit a tiny 1-second voiced sample through the full pipeline and
/// write the decoded PCM to `/tmp/g7231-sample.raw` so a human can play
/// it back with e.g. `aplay -f S16_LE -c 1 -r 8000 /tmp/g7231-sample.raw`.
/// Gated behind `#[ignore]` so normal `cargo test` runs don't touch the
/// filesystem; invoke explicitly with
/// `cargo test --release -- --ignored roundtrip_writes_sample_raw`.
#[test]
#[ignore = "writes /tmp/g7231-sample.raw; run explicitly with --ignored"]
fn roundtrip_writes_sample_raw() {
    const FRAMES: usize = 33; // ~1 s at 30 ms/frame.
    let input = voiced(FRAMES);
    let mut enc = make_encoder(Some(6300));
    enc.send_frame(&audio_frame(&input)).unwrap();
    enc.flush().unwrap();

    let mut dec = make_decoder();
    let mut decoded: Vec<u8> = Vec::with_capacity(FRAMES * FRAME_SAMPLES * 2);
    for pkt in collect_packets(&mut *enc) {
        dec.send_packet(&pkt).unwrap();
        let Frame::Audio(af) = dec.receive_frame().unwrap() else {
            panic!("expected audio frame");
        };
        decoded.extend_from_slice(&af.data[0]);
    }

    std::fs::write("/tmp/g7231-sample.raw", &decoded).expect("write /tmp/g7231-sample.raw");
    eprintln!(
        "roundtrip_writes_sample_raw: wrote {} bytes to /tmp/g7231-sample.raw",
        decoded.len()
    );
}
