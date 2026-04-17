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

use oxideav_codec::{CodecRegistry, Decoder, Encoder};
use oxideav_core::packet::PacketFlags;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Frame, Packet, Rational, Result, SampleFormat, TimeBase,
};
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
    oxideav_g7231::register(&mut reg);
    reg.make_decoder(&make_params(None)).expect("decoder ctor")
}

fn audio_frame(samples: &[i16]) -> Frame {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        format: SampleFormat::S16,
        channels: 1,
        sample_rate: SAMPLE_RATE,
        samples: samples.len() as u32,
        pts: Some(0),
        time_base: TimeBase::new(1, SAMPLE_RATE as i64),
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
                    assert_eq!(af.sample_rate, SAMPLE_RATE);
                    assert_eq!(af.channels, 1);
                    assert_eq!(af.format, SampleFormat::S16);
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
