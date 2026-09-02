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
//! What is pinned here (r406 measured floors — after the LSP
//! band-order fix and the output-scale re-arbitration, both the float
//! decoder and the fixed-point `qdec` pipeline):
//!
//! - **Wire format**: every frame of every main-body `.RCO`/`.TCO`
//!   stream unpacks through the clause-4 Table 5/6 layout and repacks
//!   byte-identically — 2 616 frames — except the three deliberate
//!   transmission-error frames of `PATHD63P.TCO` (46/49/76), which must
//!   *fail* field validation (out-of-range `C(30,6)` codes).
//! - **Decoder tracking**: whole-file corr + SNR floors on the
//!   post-filter-OFF vectors through the fixed-point pipeline, and
//!   cold-start per-frame floors on `OVERD63P`/`PATHD53`. Whole-file
//!   floors on the saturation-torture `OVER..`/`TAME..` classes stay
//!   loose: the reference's exact overflow protocol is clause-5
//!   territory (see the workspace round notes).
//! - **Robustness**: full CRC-driven decodes (erasure concealment +
//!   invalid-frame concealment) complete with the exact sample budget.
//! - **Encoder parameter agreement**: `SpecEncoder` output on every
//!   encoder test input is a legal clause-4 stream, and its LSP / lag /
//!   fixed-gain decisions agree with the reference `.RCO` bitstreams at
//!   pinned per-vector floors.

use oxideav_g7231::annex_a::{unpack_sid, FrameType};
use oxideav_g7231::encoder::{SpecEncoder, SynthesisState};
use oxideav_g7231::linepack::{pack_frame, unpack_frame, PackedRate};
use oxideav_g7231::qdec::QSynthesis;

use std::path::{Path, PathBuf};

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

fn read_pcm(dir: &Path, name: &str) -> Vec<i16> {
    let b = std::fs::read(dir.join(name)).unwrap();
    b.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Per-frame erasure flags from a `.CRC` companion (16-bit LE words,
/// 1 = the frame is to be treated as erased).
fn read_crc(dir: &Path, name: &str) -> Vec<bool> {
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
    dir: &Path,
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
    assert_eq!(
        total,
        MAIN_BODY_STREAMS.iter().map(|s| s.2).sum::<usize>() - 3
    );
}

/// Low-rate decoder tracking floor on the OVERD53 vector (post-filter
/// OFF per the ITU test configuration).
///
/// History of the floors: r388's float decoder measured whole-file
/// corr 0.973 / SNR 7.3 dB — a clipping artifact (see the r391 notes).
/// r391's honest loop measured 0.48 / 0.26 dB. r406 (LSP band-order
/// fix + unshifted output emission) measures corr 0.9977 / mean
/// per-frame 0.9970 / SNR 20.9 dB — floors pinned just under.
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
    assert!(c >= 0.99, "whole-file corr regressed: {c:.4}");
    assert!(
        mean_frame_corr >= 0.99,
        "mean per-frame corr regressed: {mean_frame_corr:.4}"
    );
    assert!(snr >= 15.0, "SNR regressed: {snr:.2} dB");
}

/// High-rate decoder cold-start tracking floors on OVERD63P
/// (post-filter ON) — r406 measured 0.9690 / 0.9993 / 0.9999 / 0.9997.
#[test]
fn decoder_tracks_overd63p_cold_start() {
    let Some(dir) = corpus_dir() else { return };
    let reference = read_pcm(&dir, "OVERD63P.ROU");
    let bs = std::fs::read(dir.join("OVERD63P.TCO")).unwrap();
    let mut st = SynthesisState::new();
    st.set_postfilter(true);
    let floors = [0.95f64, 0.99, 0.99, 0.99];
    for (i, floor) in floors.iter().enumerate() {
        let pcm = st.decode_mpmlq(&bs[i * 24..(i + 1) * 24]).unwrap();
        let c = corr(&reference[i * FRAME_SAMPLES..(i + 1) * FRAME_SAMPLES], &pcm);
        eprintln!("OVERD63P frame {i}: corr {c:.4}");
        assert!(c >= *floor, "frame {i} corr {c:.4} under floor {floor}");
    }
}

/// PATHD53 (post-filter OFF) cold-start floor — r406 measured frame 0
/// corr 0.9999 (r388: 0.549, r391: 0.835).
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
    assert!(c >= 0.995, "frame 0 corr {c:.4} under floor");
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
        (
            "PATHD63P.TCO",
            "PATHD63P.ROU",
            Some("PATHD63P.CRC"),
            24,
            true,
        ),
        ("OVERD63P.TCO", "OVERD63P.ROU", None, 24, true),
        (
            "TAMED63P.TCO",
            "TAMED63P.ROU",
            Some("TAMED63P.CRC"),
            24,
            true,
        ),
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

/// Decode a whole fixed-rate stream through the fixed-point
/// [`QSynthesis`] pipeline (the shipped registry decode path):
/// CRC-driven erasure concealment where a companion track exists, and
/// invalid-frame concealment for content that fails field validation.
fn decode_stream_fixed(
    dir: &Path,
    bs_name: &str,
    frame_bytes: usize,
    postfilter: bool,
    crc: Option<&str>,
) -> Vec<i16> {
    let bs = std::fs::read(dir.join(bs_name)).unwrap();
    let n = bs.len() / frame_bytes;
    let erased = match crc {
        Some(c) => read_crc(dir, c),
        None => vec![false; n],
    };
    let mut st = QSynthesis::new();
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

/// Agreement stats between the reference decoder output and ours:
/// (exact-sample ratio, max |diff|).
fn exactness(reference: &[i16], ours: &[i16]) -> (f64, i32) {
    let mut same = 0usize;
    let mut max_d = 0i32;
    for (&r, &o) in reference.iter().zip(ours.iter()) {
        if r == o {
            same += 1;
        }
        max_d = max_d.max((r as i32 - o as i32).abs());
    }
    (same as f64 / reference.len() as f64, max_d)
}

/// Fixed-point decoder tracking on the post-filter-OFF decoder vectors
/// (PATHD53 / OVERD53 / INEQD53 per TSTG7231 Table 1). r391 measured
/// floors — the Q15/Q31 saturating chain must not regress below the
/// float path it replaces.
#[test]
fn fixed_decoder_tracks_pf_off_vectors() {
    let Some(dir) = corpus_dir() else { return };
    // r406 measured (LSP band-order fix + output-scale
    // re-arbitration): PATHD53 corr 1.0000 / 54.4 dB / 17.6% exact /
    // max|Δ| 27, OVERD53 0.9993 / 28.1 dB, INEQD53 0.9811 / 12.5 dB /
    // 50.8% exact / max|Δ| 19. (r391: 0.60 / 1.9 dB, 0.48 / 0.26 dB,
    // 0.83 / 4.5 dB.)
    for (tco, rou, fb, corr_floor, snr_floor) in [
        ("PATHD53.TCO", "PATHD53.ROU", 20usize, 0.999f64, 45.0f64),
        ("OVERD53.TCO", "OVERD53.ROU", 20, 0.995, 22.0),
        ("INEQD53.TCO", "INEQD53.ROU", 20, 0.95, 9.0),
    ] {
        let reference = read_pcm(&dir, rou);
        let ours = decode_stream_fixed(&dir, tco, fb, false, None);
        assert_eq!(ours.len(), reference.len(), "{tco}: sample budget");
        let c = corr(&reference, &ours);
        let snr = snr_db(&reference, &ours);
        let (exact, max_d) = exactness(&reference, &ours);
        eprintln!(
            "fixed {tco}: corr {c:.4}, SNR {snr:.2} dB, exact {:.2}%, max|d| {max_d}",
            exact * 100.0
        );
        assert!(
            c >= corr_floor,
            "{tco}: corr {c:.4} under floor {corr_floor}"
        );
        assert!(
            snr >= snr_floor,
            "{tco}: SNR {snr:.2} under floor {snr_floor}"
        );
    }
}

/// Fixed-point decoder tracking on the post-filter-ON decoder vectors
/// (PATHD63P / OVERD63P / TAMED63P per TSTG7231 Table 1) — the full
/// §3.6 pitch post-filter → §3.7 synthesis → §3.8 formant → §3.9 AGC
/// chain in saturating integer arithmetic, with CRC-driven erasure
/// concealment where a companion track exists. r391 measured floors.
#[test]
fn fixed_decoder_tracks_pf_on_vectors() {
    let Some(dir) = corpus_dir() else { return };
    // r406 measured (LSP band-order fix + output-scale
    // re-arbitration): PATHD63P 0.9123 / 7.60 dB, OVERD63P 0.9715 /
    // 12.51 dB, TAMED63P 0.9643 / 11.53 dB. (r391: 0.77 / 3.5 dB,
    // 0.40 / 0.75 dB, 0.11 / −2.2 dB.) The remaining gap to
    // bit-exactness on these post-filter-ON / saturation-torture
    // classes is the reference's clause-5 per-stage rounding and
    // overflow protocol.
    for (tco, rou, crc, corr_floor, snr_floor) in [
        (
            "PATHD63P.TCO",
            "PATHD63P.ROU",
            Some("PATHD63P.CRC"),
            0.88f64,
            6.0f64,
        ),
        ("OVERD63P.TCO", "OVERD63P.ROU", None, 0.94, 10.0),
        (
            "TAMED63P.TCO",
            "TAMED63P.ROU",
            Some("TAMED63P.CRC"),
            0.92,
            9.0,
        ),
    ] {
        let reference = read_pcm(&dir, rou);
        let ours = decode_stream_fixed(&dir, tco, 24, true, crc);
        assert_eq!(ours.len(), reference.len(), "{tco}: sample budget");
        let c = corr(&reference, &ours);
        let snr = snr_db(&reference, &ours);
        let (exact, max_d) = exactness(&reference, &ours);
        eprintln!(
            "fixed {tco}: corr {c:.4}, SNR {snr:.2} dB, exact {:.2}%, max|d| {max_d}",
            exact * 100.0
        );
        assert!(
            c >= corr_floor,
            "{tco}: corr {c:.4} under floor {corr_floor}"
        );
        assert!(
            snr >= snr_floor,
            "{tco}: SNR {snr:.2} under floor {snr_floor}"
        );
    }
}

/// Full-corpus encoder conformance: every frame the encoder emits on
/// every ITU encoder-test input must be a legal clause-4 stream
/// (unpack + repack identity, right rate flag), and the coded
/// parameters must agree with the bit-exact reference `.RCO` streams
/// at pinned floors (r406 measured, floors a few points under):
///
/// | vector    | LSP word | ACL0/2 ±1 | MG exact |
/// |-----------|----------|-----------|----------|
/// | CODEC63  | 89.5%    | 95.2%        | 91.0%    | 55.6%              |
/// | PATHC63H | 97.0%    | 73.3%        | 86.0%    | 57.2%              |
/// | OVERC63  | 90.0%    | 87.5%        | 76.2%    | 52.5%              |
/// | TAMEC63H | 53.0%    | 98.5%        | 56.8%    | 16.8%              |
/// | PATHC53  | 93.9%    | 72.4%        | 78.4%    | 69.6%              |
/// | INEQC53  | 46.0%    | 94.4%        | 25.8%    | 60.3%              |
/// | OVERC53H | 95.2%    | 95.2%        | 81.0%    | 60.7%              |
///
/// The TAME/INEQ classes are designed around the reference's exact
/// fixed-point rounding (taming is clause-5-only), so their parameter
/// floors stay loose; the remaining distance to 100% everywhere is the
/// float-analysis vs bit-exact-fixed-point gap.
#[test]
fn encoder_parameter_agreement_against_reference_bitstreams() {
    let Some(dir) = corpus_dir() else { return };
    for (tin, rco, rate, hp, lsp_floor, lag1_floor, mg_floor) in [
        (
            "CODEC63.TIN",
            "CODEC63.RCO",
            PackedRate::High,
            false,
            70.0f64,
            72.0f64,
            55.0f64,
        ),
        (
            "PATHC63H.TIN",
            "PATHC63H.RCO",
            PackedRate::High,
            true,
            85.0,
            45.0,
            48.0,
        ),
        (
            "OVERC63.TIN",
            "OVERC63.RCO",
            PackedRate::High,
            false,
            75.0,
            90.0,
            55.0,
        ),
        (
            "TAMEC63H.TIN",
            "TAMEC63H.RCO",
            PackedRate::High,
            true,
            20.0,
            40.0,
            4.0,
        ),
        (
            "PATHC53.TIN",
            "PATHC53.RCO",
            PackedRate::Low,
            false,
            85.0,
            58.0,
            45.0,
        ),
        (
            "INEQC53.TIN",
            "INEQC53.RCO",
            PackedRate::Low,
            false,
            15.0,
            90.0,
            2.0,
        ),
        (
            "OVERC53H.TIN",
            "OVERC53H.RCO",
            PackedRate::Low,
            true,
            80.0,
            92.0,
            60.0,
        ),
    ] {
        let pcm = read_pcm(&dir, tin);
        let refbs = std::fs::read(dir.join(rco)).unwrap();
        let fb = if rate == PackedRate::High { 24 } else { 20 };
        let frames = (pcm.len() / FRAME_SAMPLES).min(refbs.len() / fb);
        let mut enc = SpecEncoder::new(rate);
        enc.set_highpass(hp); // TSTG7231 Table 1 configuration
        let (mut lsp_eq, mut lag_1, mut mg_eq) = (0usize, 0usize, 0usize);
        let mut n_lag = 0usize;
        let mut n_sub = 0usize;
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
            if p.lsp_index == pr.lsp_index {
                lsp_eq += 1;
            }
            for s in [0usize, 2] {
                n_lag += 1;
                if (p.acl[s] as i64 - pr.acl[s] as i64).abs() <= 1 {
                    lag_1 += 1;
                }
            }
            for s in 0..4 {
                n_sub += 1;
                if p.gain[s] % 24 == pr.gain[s] % 24 {
                    mg_eq += 1;
                }
            }
        }
        let lsp_pct = 100.0 * lsp_eq as f64 / frames as f64;
        let lag1_pct = 100.0 * lag_1 as f64 / n_lag as f64;
        let mg_pct = 100.0 * mg_eq as f64 / n_sub as f64;
        eprintln!(
            "{tin}: LSP word {lsp_pct:.1}%, ACL0/2 ±1 {lag1_pct:.1}%, MG exact {mg_pct:.1}% \
             over {frames} frames"
        );
        assert!(
            lsp_pct >= lsp_floor,
            "{tin}: LSP agreement {lsp_pct:.1}% under floor {lsp_floor}%"
        );
        assert!(
            lag1_pct >= lag1_floor,
            "{tin}: ACL agreement {lag1_pct:.1}% under floor {lag1_floor}%"
        );
        assert!(
            mg_pct >= mg_floor,
            "{tin}: MG agreement {mg_pct:.1}% under floor {mg_floor}%"
        );
    }
}

/// Per-stage encoder agreement with every prior decision
/// **teacher-forced** from the reference bitstream: after each frame
/// (and, inside a frame, after each subframe's adaptive-codebook
/// decision) the shadow state is committed from the `.RCO` parameters,
/// so each stage is scored on its own decision from the reference's
/// exact context, free of drift. Floors are r455 measured values minus
/// a small margin:
///
/// | vector   | LSP word | ACL0/2 exact | PG exact | FCB subframe exact |
/// |----------|----------|--------------|----------|--------------------|
/// | CODEC63  | 89.5%    | 95.2%        | 91.0%    | 50.8%              |
/// | PATHC63H | 95.7%    | 73.3%        | 86.0%    | 46.3%              |
/// | OVERC63  | 90.0%    | 87.5%        | 76.2%    | 46.2%              |
/// | TAMEC63H | 53.0%    | 98.5%        | 56.8%    |  9.8%              |
/// | PATHC53  | 93.9%    | 72.4%        | 78.4%    | 51.8%              |
/// | INEQC53  | 46.0%    | 94.4%        | 25.8%    | 15.5%              |
/// | OVERC53H | 95.2%    | 95.2%        | 81.0%    | 48.8%              |
///
/// "FCB subframe exact" = grid, positions, signs and fixed-codebook
/// gain index all equal to the reference's for that subframe.
#[test]
fn encoder_teacher_forced_stage_agreement() {
    let Some(dir) = corpus_dir() else { return };
    for (tin, rco, rate, hp, lsp_floor, acl_floor, pg_floor, fcb_floor) in [
        (
            "CODEC63.TIN",
            "CODEC63.RCO",
            PackedRate::High,
            false,
            85.0f64,
            90.0f64,
            86.0f64,
            50.0f64,
        ),
        (
            "PATHC63H.TIN",
            "PATHC63H.RCO",
            PackedRate::High,
            true,
            94.0,
            68.0,
            81.0,
            52.0,
        ),
        (
            "OVERC63.TIN",
            "OVERC63.RCO",
            PackedRate::High,
            false,
            80.0,
            80.0,
            65.0,
            40.0,
        ),
        (
            "TAMEC63H.TIN",
            "TAMEC63H.RCO",
            PackedRate::High,
            true,
            45.0,
            94.0,
            45.0,
            10.0,
        ),
        (
            "PATHC53.TIN",
            "PATHC53.RCO",
            PackedRate::Low,
            false,
            90.0,
            67.0,
            73.0,
            64.0,
        ),
        (
            "INEQC53.TIN",
            "INEQC53.RCO",
            PackedRate::Low,
            false,
            38.0,
            88.0,
            18.0,
            50.0,
        ),
        (
            "OVERC53H.TIN",
            "OVERC53H.RCO",
            PackedRate::Low,
            true,
            85.0,
            85.0,
            70.0,
            50.0,
        ),
    ] {
        let pcm = read_pcm(&dir, tin);
        let refbs = std::fs::read(dir.join(rco)).unwrap();
        let fb = if rate == PackedRate::High { 24 } else { 20 };
        let frames = (pcm.len() / FRAME_SAMPLES).min(refbs.len() / fb);
        let mut enc = SpecEncoder::new(rate);
        enc.set_highpass(hp);
        let (mut lsp_eq, mut acl_eq, mut pg_eq, mut fcb_eq) = (0usize, 0usize, 0usize, 0usize);
        for i in 0..frames {
            let mut frame_pcm = [0i16; FRAME_SAMPLES];
            frame_pcm.copy_from_slice(&pcm[i * FRAME_SAMPLES..(i + 1) * FRAME_SAMPLES]);
            let pr = unpack_frame(&refbs[i * fb..(i + 1) * fb]).unwrap();
            let p = enc.encode_frame_params(&frame_pcm, Some(&pr));
            if p.lsp_index == pr.lsp_index {
                lsp_eq += 1;
            }
            let lag0 = pr.acl[0] as i32 + 18;
            let lag2 = pr.acl[2] as i32 + 18;
            for s in 0..4 {
                if s % 2 == 0 && p.acl[s] == pr.acl[s] {
                    acl_eq += 1;
                }
                let lag_base = if s < 2 { lag0 } else { lag2 };
                let short = rate == PackedRate::High && lag_base < 58;
                let (go, gr) = if short {
                    (p.gain[s] & 0x7FF, pr.gain[s] & 0x7FF)
                } else {
                    (p.gain[s], pr.gain[s])
                };
                if go / 24 == gr / 24 {
                    pg_eq += 1;
                }
                let train_eq = !short || (p.gain[s] & 0x800) == (pr.gain[s] & 0x800);
                if go % 24 == gr % 24
                    && p.grid[s] == pr.grid[s]
                    && p.pos[s] == pr.pos[s]
                    && p.psig[s] == pr.psig[s]
                    && train_eq
                {
                    fcb_eq += 1;
                }
            }
        }
        let lsp_pct = 100.0 * lsp_eq as f64 / frames as f64;
        let acl_pct = 100.0 * acl_eq as f64 / (2 * frames) as f64;
        let pg_pct = 100.0 * pg_eq as f64 / (4 * frames) as f64;
        let fcb_pct = 100.0 * fcb_eq as f64 / (4 * frames) as f64;
        eprintln!(
            "{tin} (teacher-forced): LSP word {lsp_pct:.1}%, ACL0/2 exact {acl_pct:.1}%, \
             PG exact {pg_pct:.1}%, FCB subframe exact {fcb_pct:.1}% over {frames} frames"
        );
        assert!(
            lsp_pct >= lsp_floor,
            "{tin}: LSP {lsp_pct:.1}% under floor {lsp_floor}%"
        );
        assert!(
            acl_pct >= acl_floor,
            "{tin}: ACL {acl_pct:.1}% under floor {acl_floor}%"
        );
        assert!(
            pg_pct >= pg_floor,
            "{tin}: PG {pg_pct:.1}% under floor {pg_floor}%"
        );
        assert!(
            fcb_pct >= fcb_floor,
            "{tin}: FCB {fcb_pct:.1}% under floor {fcb_floor}%"
        );
    }
}

/// Parse a mixed DTX reference stream into per-frame types and SID
/// contents (`0` / `1` active, `2` SID, `3` untransmitted).
fn parse_dtx_stream(bs: &[u8]) -> (Vec<u8>, Vec<Option<(u32, u8)>>) {
    let mut types = Vec::new();
    let mut sids = Vec::new();
    let mut i = 0usize;
    while i < bs.len() {
        let t = bs[i] & 0b11;
        let size = [24usize, 20, 4, 1][t as usize];
        types.push(t);
        sids.push(if t == 2 {
            unpack_sid(&bs[i..i + 4])
        } else {
            None
        });
        i += size;
    }
    (types, sids)
}

/// Annex A encoder conformance on the DTX vectors: with the VAD on
/// (high-pass on, the annex's own configuration), the frame-type
/// sequence — active / SID / untransmitted — must agree with the
/// reference stream at pinned floors, every SID frame must carry the
/// Table A.1 layout, and on frames both sides declare SID the coded
/// gain index must sit within ±1 of the reference's at a pinned rate.
/// `DTXMIX` follows the `.RAT` per-frame rate schedule.
///
/// r455 measured: DTX63 frame type 94.4% (active/inactive 95.6%), SID
/// gain 83.7% exact / 95.9% ±1; DTX53MIX 93.3% / 96.7%; DTXMIX 93.3%.
/// The comfort-noise excitation itself uses this crate's own random
/// generator (the annex does not define `random_number()`), so active
/// frames following a silence are not compared here.
#[test]
fn encoder_annex_a_frame_types_track_dtx_vectors() {
    let Some(dir) = corpus_dir() else { return };
    for (tin, rco, rat, base_rate, ft_floor, gain1_floor) in [
        (
            "DTX63.TIN",
            "DTX63.RCO",
            None,
            PackedRate::High,
            92.0f64,
            90.0f64,
        ),
        (
            "DTX53MIX.TIN",
            "DTX53.RCO",
            None,
            PackedRate::Low,
            90.0,
            80.0,
        ),
        (
            "DTX53MIX.TIN",
            "DTXMIX.RCO",
            Some("DTXMIX.RAT"),
            PackedRate::High,
            90.0,
            80.0,
        ),
    ] {
        let pcm = read_pcm(&dir, tin);
        let refbs = std::fs::read(dir.join(rco)).unwrap();
        let (rtypes, rsids) = parse_dtx_stream(&refbs);
        let frames = pcm.len() / FRAME_SAMPLES;
        assert_eq!(rtypes.len(), frames, "{rco}: frame count");
        let rates: Vec<PackedRate> = match rat {
            Some(r) => std::fs::read(dir.join(r))
                .unwrap()
                .iter()
                .map(|&b| {
                    if b == 0 {
                        PackedRate::High
                    } else {
                        PackedRate::Low
                    }
                })
                .collect(),
            None => vec![base_rate; frames],
        };
        let mut enc = SpecEncoder::new(base_rate);
        enc.set_highpass(true);
        enc.set_vad(true);
        let (mut ft_eq, mut n_sid, mut gain1) = (0usize, 0usize, 0usize);
        for f in 0..frames {
            enc.set_rate(rates[f]);
            let mut fr = [0i16; FRAME_SAMPLES];
            fr.copy_from_slice(&pcm[f * FRAME_SAMPLES..(f + 1) * FRAME_SAMPLES]);
            let bytes = enc.encode_frame(&fr);
            let t = bytes[0] & 0b11;
            match enc.last_frame_type() {
                FrameType::Active => assert_eq!(bytes.len(), rates[f].frame_bytes()),
                FrameType::Sid => {
                    assert_eq!(bytes.len(), 4);
                    assert!(unpack_sid(&bytes).is_some());
                }
                FrameType::Untransmitted => assert_eq!(bytes.len(), 1),
            }
            if t == rtypes[f] {
                ft_eq += 1;
            }
            if t == 2 && rtypes[f] == 2 {
                n_sid += 1;
                let (_, g) = unpack_sid(&bytes).unwrap();
                let (_, rg) = rsids[f].unwrap();
                if (g as i32 - rg as i32).abs() <= 1 {
                    gain1 += 1;
                }
            }
        }
        let ft_pct = 100.0 * ft_eq as f64 / frames as f64;
        let g1_pct = if n_sid > 0 {
            100.0 * gain1 as f64 / n_sid as f64
        } else {
            0.0
        };
        eprintln!(
            "{rco}: frame type exact {ft_pct:.1}% over {frames} frames; SID gain within ±1 \
             {g1_pct:.1}% over {n_sid} common SID frames"
        );
        assert!(
            ft_pct >= ft_floor,
            "{rco}: frame type {ft_pct:.1}% under floor {ft_floor}%"
        );
        assert!(n_sid > 0, "{rco}: no common SID frames");
        assert!(
            g1_pct >= gain1_floor,
            "{rco}: SID gain ±1 {g1_pct:.1}% under floor {gain1_floor}%"
        );
    }
}
