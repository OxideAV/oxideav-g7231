//! ITU-T G.723.1 conformance-vector harness.
//!
//! Runs against the official CNET/France Télécom digital test sequences
//! staged under `docs/audio/g7231/conformance/` in the OxideAV umbrella
//! (black-box input → output pairs; see that directory's README for the
//! class/name/extension taxonomy). The corpus is not shipped with the
//! crate: when the directory is absent (standalone checkout, per-crate
//! CI) every test here logs a skip notice and passes vacuously.
//!
//! Override the corpus location with `OXIDEAV_G7231_CONFORMANCE`.
//!
//! What is pinned here (r388 measured floors, floating-point decoder):
//!
//! - **Wire format**: every frame of every main-body `.RCO`/`.TCO`
//!   stream unpacks through the clause-4 Table 5/6 layout and repacks
//!   byte-identically — 2 616 frames — except the three deliberate
//!   transmission-error frames of `PATHD63P.TCO` (46/49/76), which must
//!   *fail* field validation (out-of-range `C(30,6)` codes).
//! - **Decoder tracking**: whole-file waveform correlation ≥ 0.95 on
//!   `OVERD53` (post-filter OFF) and cold-start per-frame correlation
//!   floors on `OVERD63P` (post-filter ON). The `OVER..`/`TAME..`
//!   classes deliberately drive sustained Word16-saturation chains that
//!   only a bit-exact fixed-point implementation tracks long-range, so
//!   whole-file floors on those streams are intentionally not pinned.
//! - **Robustness**: full CRC-driven decodes (erasure concealment +
//!   invalid-frame concealment) complete with the exact sample budget.
//! - **Encoder self-validity**: `SpecEncoder` output on the encoder
//!   test inputs is a legal clause-4 stream at both rates.

use oxideav_g7231::encoder::{SpecEncoder, SynthesisState};
use oxideav_g7231::linepack::{pack_frame, unpack_frame, PackedRate};

use std::path::PathBuf;

const FRAME_SAMPLES: usize = 240;

fn corpus_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("OXIDEAV_G7231_CONFORMANCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/audio/g7231/conformance")
        });
    if dir.join("PATHC63H.RCO").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "skipping ITU conformance test: corpus not found at {} \
             (set OXIDEAV_G7231_CONFORMANCE)",
            dir.display()
        );
        None
    }
}

fn read_pcm(dir: &PathBuf, name: &str) -> Vec<i16> {
    let b = std::fs::read(dir.join(name)).unwrap();
    b.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Per-frame erasure flags from a `.CRC` companion (16-bit LE words,
/// 1 = the frame is to be treated as erased).
fn read_crc(dir: &PathBuf, name: &str) -> Vec<bool> {
    let b = std::fs::read(dir.join(name)).unwrap();
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]) != 0)
        .collect()
}

/// Decode a whole fixed-rate ITU bitstream file with the given
/// post-filter configuration and optional CRC erasure track;
/// content-invalid frames conceal like erasures (the decoder-vector
/// contract established by the PATHD63P transmission-error frames).
fn decode_stream(
    dir: &PathBuf,
    bs_name: &str,
    frame_bytes: usize,
    postfilter: bool,
    crc: Option<&str>,
) -> Vec<i16> {
    let bs = std::fs::read(dir.join(bs_name)).unwrap();
    assert_eq!(
        bs.len() % frame_bytes,
        0,
        "{bs_name}: stream must be whole frames"
    );
    let n = bs.len() / frame_bytes;
    let erased = match crc {
        Some(c) => read_crc(dir, c),
        None => vec![false; n],
    };
    assert_eq!(erased.len(), n, "{bs_name}: CRC track length mismatch");
    let mut st = SynthesisState::new();
    st.set_postfilter(postfilter);
    let mut out = Vec::with_capacity(n * FRAME_SAMPLES);
    for i in 0..n {
        let frame = &bs[i * frame_bytes..(i + 1) * frame_bytes];
        let pcm = if erased[i] {
            st.decode_erased()
        } else if frame_bytes == 24 {
            st.decode_mpmlq(frame)
                .unwrap_or_else(|_| st.decode_erased())
        } else {
            st.decode_acelp(frame)
                .unwrap_or_else(|_| st.decode_erased())
        };
        out.extend_from_slice(&pcm);
    }
    out
}

/// Normalised cross-correlation of one already-aligned window.
fn corr(a: &[i16], b: &[i16]) -> f64 {
    let mut dot = 0f64;
    let mut ea = 0f64;
    let mut eb = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        ea += (x as f64).powi(2);
        eb += (y as f64).powi(2);
    }
    if ea > 0.0 && eb > 0.0 {
        dot / (ea.sqrt() * eb.sqrt())
    } else {
        0.0
    }
}

fn snr_db(reference: &[i16], ours: &[i16]) -> f64 {
    let mut sig = 0f64;
    let mut err = 0f64;
    for (&r, &o) in reference.iter().zip(ours.iter()) {
        sig += (r as f64).powi(2);
        err += (r as f64 - o as f64).powi(2);
    }
    if err > 0.0 {
        10.0 * (sig / err).log10()
    } else {
        f64::INFINITY
    }
}

/// The 13 main-body bitstream files (name, frame bytes, frame count per
/// TSTG7231.DOC Tables 2–6).
const MAIN_BODY_STREAMS: [(&str, usize, usize); 13] = [
    ("PATHC63H.RCO", 24, 1019),
    ("PATHC53.RCO", 20, 1028),
    ("CODEC63.RCO", 24, 313),
    ("INEQC53.RCO", 20, 63),
    ("OVERC53H.RCO", 20, 21),
    ("OVERC63.RCO", 24, 20),
    ("TAMEC63H.RCO", 24, 100),
    ("PATHD63P.TCO", 24, 100),
    ("PATHD53.TCO", 20, 4),
    ("OVERD63P.TCO", 24, 33),
    ("OVERD53.TCO", 20, 26),
    ("INEQD53.TCO", 20, 2),
    ("TAMED63P.TCO", 24, 100),
];

/// Frames of PATHD63P.TCO carrying deliberate transmission errors
/// (out-of-range combinatorial position codes). They are NOT flagged in
/// PATHD63P.CRC — the decoder discovers them via field validation.
const PATHD63P_INVALID_FRAMES: [usize; 3] = [46, 49, 76];

/// Every frame of every main-body ITU stream must round-trip through
/// the clause-4 unpack → repack path byte-identically; the only
/// exceptions are PATHD63P's three transmission-error frames, which
/// must be *rejected* by field validation.
#[test]
fn itu_streams_unpack_and_repack_byte_identically() {
    let Some(dir) = corpus_dir() else { return };
    let mut total = 0usize;
    for (name, fb, expect_frames) in MAIN_BODY_STREAMS {
        let bs = std::fs::read(dir.join(name)).unwrap();
        assert_eq!(bs.len(), fb * expect_frames, "{name}: size per TSTG7231");
        for i in 0..expect_frames {
            let frame = &bs[i * fb..(i + 1) * fb];
            let is_error_frame = name == "PATHD63P.TCO" && PATHD63P_INVALID_FRAMES.contains(&i);
            match unpack_frame(frame) {
                Ok(p) => {
                    assert!(
                        !is_error_frame,
                        "{name} frame {i}: transmission-error frame must fail validation"
                    );
                    let re = pack_frame(&p).unwrap();
                    assert_eq!(re.as_slice(), frame, "{name} frame {i}: repack identity");
                    total += 1;
                }
                Err(e) => {
                    assert!(
                        is_error_frame,
                        "{name} frame {i}: unexpected unpack failure: {e}"
                    );
                }
            }
        }
    }
    // 2 816 frames total minus the 3 error frames.
    assert_eq!(total, MAIN_BODY_STREAMS.iter().map(|s| s.2).sum::<usize>() - 3);
}

/// Low-rate decoder tracking floor on the OVERD53 vector (post-filter
/// OFF per the ITU test configuration). r388 measured: whole-file corr
/// 0.973, mean per-frame corr 0.985, SNR 7.3 dB.
#[test]
fn decoder_tracks_overd53_reference() {
    let Some(dir) = corpus_dir() else { return };
    let reference = read_pcm(&dir, "OVERD53.ROU");
    let ours = decode_stream(&dir, "OVERD53.TCO", 20, false, None);
    assert_eq!(ours.len(), reference.len());
    let c = corr(&reference, &ours);
    let mut frame_corr_sum = 0f64;
    let n = ours.len() / FRAME_SAMPLES;
    for f in 0..n {
        frame_corr_sum += corr(
            &reference[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES],
            &ours[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES],
        );
    }
    let mean_frame_corr = frame_corr_sum / n as f64;
    let snr = snr_db(&reference, &ours);
    eprintln!("OVERD53: corr {c:.4}, mean frame corr {mean_frame_corr:.4}, SNR {snr:.2} dB");
    assert!(c >= 0.95, "whole-file corr regressed: {c:.4}");
    assert!(
        mean_frame_corr >= 0.97,
        "mean per-frame corr regressed: {mean_frame_corr:.4}"
    );
    assert!(snr >= 5.0, "SNR regressed: {snr:.2} dB");
}

/// High-rate decoder cold-start tracking floors on OVERD63P
/// (post-filter ON). r388 measured frames 0–3: 0.605 / 0.887 / 0.833 /
/// 0.738. Long-range tracking of the OVER class needs bit-exact
/// fixed-point saturation and is not pinned.
#[test]
fn decoder_tracks_overd63p_cold_start() {
    let Some(dir) = corpus_dir() else { return };
    let reference = read_pcm(&dir, "OVERD63P.ROU");
    let bs = std::fs::read(dir.join("OVERD63P.TCO")).unwrap();
    let mut st = SynthesisState::new();
    st.set_postfilter(true);
    let floors = [0.5f64, 0.8, 0.75, 0.65];
    for (i, floor) in floors.iter().enumerate() {
        let pcm = st.decode_mpmlq(&bs[i * 24..(i + 1) * 24]).unwrap();
        let c = corr(
            &reference[i * FRAME_SAMPLES..(i + 1) * FRAME_SAMPLES],
            &pcm,
        );
        eprintln!("OVERD63P frame {i}: corr {c:.4}");
        assert!(c >= *floor, "frame {i} corr {c:.4} under floor {floor}");
    }
}

/// PATHD53 (post-filter OFF) cold-start floor — r388 measured frame 0
/// corr 0.549.
#[test]
fn decoder_tracks_pathd53_cold_start() {
    let Some(dir) = corpus_dir() else { return };
    let reference = read_pcm(&dir, "PATHD53.ROU");
    let bs = std::fs::read(dir.join("PATHD53.TCO")).unwrap();
    let mut st = SynthesisState::new();
    st.set_postfilter(false);
    let pcm = st.decode_acelp(&bs[..20]).unwrap();
    let c = corr(&reference[..FRAME_SAMPLES], &pcm);
    eprintln!("PATHD53 frame 0: corr {c:.4}");
    assert!(c >= 0.45, "frame 0 corr {c:.4} under floor");
}

/// Every decoder-test stream must decode end-to-end (CRC-driven
/// erasure concealment where a `.CRC` companion exists, invalid-frame
/// concealment for the PATHD63P transmission errors) to exactly the
/// reference sample budget, with reported agreement so regressions are
/// visible in the log.
#[test]
fn decoder_full_runs_complete_with_exact_sample_budget() {
    let Some(dir) = corpus_dir() else { return };
    for (tco, rou, crc, fb, pf) in [
        ("PATHD63P.TCO", "PATHD63P.ROU", Some("PATHD63P.CRC"), 24, true),
        ("OVERD63P.TCO", "OVERD63P.ROU", None, 24, true),
        ("TAMED63P.TCO", "TAMED63P.ROU", Some("TAMED63P.CRC"), 24, true),
        ("PATHD53.TCO", "PATHD53.ROU", None, 20, false),
        ("OVERD53.TCO", "OVERD53.ROU", None, 20, false),
        ("INEQD53.TCO", "INEQD53.ROU", None, 20, false),
    ] {
        let reference = read_pcm(&dir, rou);
        let ours = decode_stream(&dir, tco, fb, pf, crc);
        assert_eq!(ours.len(), reference.len(), "{tco}: sample budget");
        eprintln!(
            "{tco}: corr {:.4}, SNR {:.2} dB over {} samples",
            corr(&reference, &ours),
            snr_db(&reference, &ours),
            ours.len()
        );
    }
}

/// The encoder must emit legal clause-4 streams on the ITU encoder-test
/// inputs (HP OFF configurations, per TSTG7231 Table 1): every frame
/// unpacks, carries the right rate flag, and repacks identically.
/// Field-level agreement against the fixed-point reference `.RCO` is
/// reported (not floored — parameter decisions of a floating-point
/// analysis pipeline diverge from the bit-exact reference).
#[test]
fn encoder_emits_self_valid_streams_on_itu_inputs() {
    let Some(dir) = corpus_dir() else { return };
    for (tin, rco, rate, fb, frames) in [
        ("CODEC63.TIN", "CODEC63.RCO", PackedRate::High, 24usize, 40usize),
        ("PATHC53.TIN", "PATHC53.RCO", PackedRate::Low, 20, 40),
    ] {
        let pcm = read_pcm(&dir, tin);
        let refbs = std::fs::read(dir.join(rco)).unwrap();
        let mut enc = SpecEncoder::new(rate);
        enc.set_highpass(false); // TSTG7231 Table 1: these run HP OFF
        let mut lag_close = 0usize;
        for i in 0..frames {
            let mut frame_pcm = [0i16; FRAME_SAMPLES];
            frame_pcm.copy_from_slice(&pcm[i * FRAME_SAMPLES..(i + 1) * FRAME_SAMPLES]);
            let bytes = enc.encode_frame(&frame_pcm);
            assert_eq!(bytes.len(), fb, "{tin} frame {i}: frame size");
            let p = unpack_frame(&bytes).unwrap_or_else(|e| {
                panic!("{tin} frame {i}: our encoder emitted an invalid frame: {e}")
            });
            assert_eq!(p.rate, rate, "{tin} frame {i}: rate flag");
            assert_eq!(
                pack_frame(&p).unwrap(),
                bytes,
                "{tin} frame {i}: repack identity"
            );
            let pr = unpack_frame(&refbs[i * fb..(i + 1) * fb]).unwrap();
            if (p.acl[0] as i64 - pr.acl[0] as i64).abs() <= 2 {
                lag_close += 1;
            }
        }
        eprintln!("{tin}: {lag_close}/{frames} frames with ACL0 within ±2 of the reference");
    }
}
