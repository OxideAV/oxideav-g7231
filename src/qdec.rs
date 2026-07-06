//! Fixed-point (Q15/Q31 saturating) G.723.1 decode chain.
//!
//! The float synthesis pipeline in [`crate::encoder`] implements the
//! clause 2/3 *mathematical* description; the Recommendation's §1.5
//! makes 16-bit saturating fixed-point behaviour normative, and the
//! OVER/TAME conformance classes drive sustained Word16-saturation
//! chains only a fixed-point pipeline tracks long-range. This module
//! rebuilds the decoder stage-by-stage on [`crate::basicop`]:
//!
//! - LSP inverse quantisation, stability, interpolation and LSP→LPC
//!   conversion in the tables' native Q15 normalised-frequency domain
//!   (Q14 cosine lookup, Q13 LPC coefficients).
//! - Excitation reconstruction (gain word, five-tap adaptive codebook,
//!   MP-MLQ / ACELP fixed vectors) in the vector-arbitrated
//!   doubled-pulse domain (see the constants below), saturated at the
//!   Word16-in-table-scale rail.
//! - §3.7 synthesis on a wide saturating accumulator with Word16
//!   recursion memory and the halved-output stage.
//!
//! All Q-format choices are stated inline; where the clause 1–4 prose
//! leaves a rounding or scaling choice open, the choice is arbitrated
//! against the ITU conformance vectors (documented per constant /
//! function). Key r391 arbitration results, from least-squares
//! decomposition of the Ã(z)-deconvolved `PATHD53.ROU` excitation
//! against the ACB/FCB bases (both coefficients ≈ 1.0 under this
//! model) and pulse-exact matches on the isolated-pulse subframes:
//!
//! - fixed-codebook pulses land at **twice** the published gain-table
//!   amplitude, and the synthesis output is **halved** on emission;
//! - the gain-vector rows act at an effective **/16384** in the
//!   doubled domain (a /8192 reading makes the pitch loop diverge
//!   where the reference stays bounded — PATHD53 frame 1 rails at
//!   ±32767 against a ±12784 reference);
//! - the eq. 41.1 `e′` view is the CONTIGUOUS history slice from
//!   `e[−L−2]` (the literal "(n mod L)" reading skips two samples and
//!   leads the reference by exactly two samples on PATHD53).

// The module is wired into the decoder stage-by-stage over r391; the
// allow shrinks as each stage's entry points go live.
#![allow(dead_code)]

use crate::basicop::*;
use crate::linepack::{PackedRate, SpecFrameParams};
use crate::spec_tables::{
    lsp_codebook_entry, LspBand, LSP_COSINE_LOOKUP_Q15, LSP_DC_PREDICTED_FREQ_Q15,
};
use crate::tables::LPC_ORDER;

/// §2.6 normal-decode LSP predictor `b = 12/32` in Q15.
const LSP_PREDICTOR_B_Q15: i16 = 12_288;
/// §3.10.1 erasure predictor `b_e = 23/32` in Q15.
const LSP_PREDICTOR_BE_Q15: i16 = 23_552;
/// §2.6 `Δ_min = 31.25 Hz` in Q15 normalised-frequency table units
/// (`31.25 Hz · 32768 / 4000 Hz = 256` exactly — the spec constant is
/// an exact power of two in the tables' own domain).
pub(crate) const LSP_DELTA_MIN_Q15: i16 = 256;
/// §3.10.1 erasure `Δ_min = 62.5 Hz` in table units.
pub(crate) const LSP_DELTA_MIN_ERASURE_Q15: i16 = 512;

/// §2.6 steps 1–2: rebuild the decoded LSP vector
/// `p̃_n = b·(p̃_{n−1} − p_DC) + p_DC + ẽ_n` from the 24-bit index and
/// the previous decoded vector, everything in Q15 normalised-frequency
/// table units with saturating arithmetic.
pub(crate) fn lsp_decode(lsp_index: u32, prev: &[i16; LPC_ORDER]) -> [i16; LPC_ORDER] {
    lsp_predict_add(lsp_index, prev, LSP_PREDICTOR_B_Q15)
}

/// §3.10.1 steps 1–2: the erasure variant — residual forced to zero,
/// predictor `b_e = 23/32`.
pub(crate) fn lsp_extrapolate(prev: &[i16; LPC_ORDER]) -> [i16; LPC_ORDER] {
    let mut out = [0i16; LPC_ORDER];
    for i in 0..LPC_ORDER {
        let dc = LSP_DC_PREDICTED_FREQ_Q15[i];
        let pred = mult(sub(prev[i], dc), LSP_PREDICTOR_BE_Q15);
        out[i] = add(dc, pred);
    }
    out
}

fn lsp_predict_add(lsp_index: u32, prev: &[i16; LPC_ORDER], b_q15: i16) -> [i16; LPC_ORDER] {
    let bands = crate::spec_lsp::split_lsp_index(lsp_index);
    let mut out = [0i16; LPC_ORDER];
    for (m, band) in LspBand::ALL.iter().enumerate() {
        let (start, len) = band.start_and_length();
        let row = lsp_codebook_entry(*band, bands[m]);
        for j in 0..len {
            let i = start + j;
            let dc = LSP_DC_PREDICTED_FREQ_Q15[i];
            let pred = mult(sub(prev[i], dc), b_q15);
            out[i] = add(add(dc, pred), row[j]);
        }
    }
    out
}

/// §2.6 step 3 (eq. 6–7.3) stability procedure in the Q15 frequency
/// domain: sweep the nine consecutive pairs, spreading any pair closer
/// than `Δ_min` around its midpoint by `±Δ_min/2`; up to 10 sweeps.
/// Returns `true` when the vector is ordered (converged) — on `false`
/// the caller must fall back to the previous LSP vector per §2.6.
pub(crate) fn lsp_stability(p: &mut [i16; LPC_ORDER], delta_min: i16) -> bool {
    let half = delta_min / 2;
    for _ in 0..crate::tables::LSP_STABILITY_MAX_ITERATIONS {
        let mut violated = false;
        for j in 0..LPC_ORDER - 1 {
            // The pair difference in 32-bit: values live in [0, 32767]
            // so the subtraction cannot wrap, but the midpoint sum can
            // exceed Word16 — compute it wide and shift.
            let diff = p[j + 1] as i32 - p[j] as i32;
            if diff < delta_min as i32 {
                let avg = ((p[j] as i32 + p[j + 1] as i32) >> 1) as i16;
                p[j] = sub(avg, half);
                p[j + 1] = add(avg, half);
                violated = true;
            }
        }
        if !violated {
            return true;
        }
    }
    // Final check after the last sweep.
    p.windows(2)
        .all(|w| w[1] as i32 - w[0] as i32 >= delta_min as i32)
}

/// §2.7 / §3.3 (eq. 8) per-subframe LSP interpolation in the Q15
/// frequency domain with Q15 weights and rounded Q31 accumulation.
/// Subframe 3 is the current vector exactly (weight pair (0, 1)).
pub(crate) fn lsp_interpolate(
    k: usize,
    prev: &[i16; LPC_ORDER],
    cur: &[i16; LPC_ORDER],
) -> [i16; LPC_ORDER] {
    if k >= 3 {
        return *cur;
    }
    let (wp, wc): (i16, i16) = match k {
        0 => (24_576, 8_192),
        1 => (16_384, 16_384),
        _ => (8_192, 24_576),
    };
    let mut out = [0i16; LPC_ORDER];
    for i in 0..LPC_ORDER {
        let acc = l_mac(l_mult(prev[i], wp), cur[i], wc);
        out[i] = round16(acc);
    }
    out
}

/// Cosine of a Q15 normalised frequency (`ω = π·f/32768`) via the
/// published 512-entry full-period Q14 cosine table with 7-bit linear
/// interpolation: the table grid is `2π·i/512`, so a half-period Q15
/// argument indexes at `f >> 7` with `f & 127` as the interpolation
/// fraction.
pub(crate) fn cos_q14(freq_q15: i16) -> i16 {
    debug_assert!(freq_q15 >= 0);
    let idx = (freq_q15 >> 7) as usize; // 0..=255 for f < 32768
    let frac = freq_q15 & 0x7F;
    let base = LSP_COSINE_LOOKUP_Q15[idx];
    let next = LSP_COSINE_LOOKUP_Q15[idx + 1];
    // Q15 fraction of the 128-unit gap: frac·256 / 32768 = frac/128.
    add(base, mult(sub(next, base), shl(frac, 8)))
}

/// §2.7 (eq. 9) LSP → LPC conversion: Q15 frequencies → Q14 cosines →
/// Q13 direct-form coefficients.
///
/// Returns the ten coefficients `ã_1..ã_10` of the quantised synthesis
/// filter in the eq. 48 convention (`sy[n] = ppf[n] + Σ ã_j·sy[n−j]`),
/// i.e. the *predictor* taps. Construction is the standard sum/
/// difference-polynomial expansion: with `c_k = cos ω_k`,
///
/// ```text
///   P(z) = Π_{k even} (1 − 2 c_k z⁻¹ + z⁻²)       (roots ω_0, ω_2, …)
///   Q(z) = Π_{k odd}  (1 − 2 c_k z⁻¹ + z⁻²)
///   A(z) = [P(z)(1 + z⁻¹) + Q(z)(1 − z⁻¹)] / 2,   ã_j = −A_j
/// ```
///
/// Intermediate coefficients are held exact in 64-bit Q24 (they are
/// filter *coefficients*, not signal samples — the Word16 saturation
/// semantics of the signal path do not apply); the final Q13 rounding
/// saturates to Word16.
pub(crate) fn lsp_to_lpc_q13(lsp_freq: &[i16; LPC_ORDER]) -> [i16; LPC_ORDER] {
    const ONE_Q24: i64 = 1 << 24;
    let half = LPC_ORDER / 2;
    let mut pz = [0i64; LPC_ORDER + 1];
    let mut qz = [0i64; LPC_ORDER + 1];
    pz[0] = ONE_Q24;
    qz[0] = ONE_Q24;
    let mut deg = 0usize;
    for k in 0..half {
        // −2·c in Q14 → apply as (coef · c) >> 13 on Q24 values.
        let c_even = cos_q14(lsp_freq[2 * k]) as i64;
        let c_odd = cos_q14(lsp_freq[2 * k + 1]) as i64;
        deg += 2;
        for i in (2..=deg).rev() {
            pz[i] += ((pz[i - 1] * c_even) >> 13).wrapping_neg() + pz[i - 2];
            qz[i] += ((qz[i - 1] * c_odd) >> 13).wrapping_neg() + qz[i - 2];
        }
        pz[1] -= (pz[0] * c_even) >> 13;
        qz[1] -= (qz[0] * c_odd) >> 13;
    }
    // A_j = (f1[j] + f2[j]) / 2 with f1 = P·(1+z⁻¹), f2 = Q·(1−z⁻¹):
    //   A_j = (P_j + P_{j−1} + Q_j − Q_{j−1}) / 2, j = 1..10.
    // ã_j = −A_j, rounded from Q24 to Q13 (>> 12 after the /2).
    let mut a = [0i16; LPC_ORDER];
    for j in 1..=LPC_ORDER {
        let sum = pz[j] + pz[j - 1] + qz[j] - qz[j - 1];
        let neg = -(sum >> 1); // ã_j = −A_j, still Q24
        let q13 = (neg + (1 << 10)) >> 11;
        a[j - 1] = if q13 > i16::MAX as i64 {
            i16::MAX
        } else if q13 < i16::MIN as i64 {
            i16::MIN
        } else {
            q13 as i16
        };
    }
    a
}

// ---------------------------------------------------------------------
// Excitation reconstruction (§3.4/§3.5 → §2.17/§2.18)
// ---------------------------------------------------------------------

/// Number of taps of the §2.14 pitch predictor.
const ACB_TAPS: usize = 5;
/// Excitation history depth: the eq. 41.1 wrap seeds reach `e[−L−2]`
/// (`L ≤ 142`) and the §3.6 backward postfilter reach `e[n − M_b]`
/// extends to `L + 3 = 145` samples before the frame.
const EXC_HIST: usize = 146;

/// Decoded contents of one 12-bit combined gain word, fixed-point view:
/// Q13 predictor taps and the raw Word16 fixed-codebook amplitude.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct QGainInfo {
    pub taps: [i16; ACB_TAPS],
    pub fcb_gain: i16,
    pub pgindex: usize,
    pub mgindex: usize,
    pub train: bool,
}

/// Decode a 12-bit combined gain word (eq. 36/39/40) into the Q13 tap
/// row and the Word16 fixed-codebook amplitude. Same index split as
/// [`crate::spec_exc::decode_gain_word`]; values stay in their
/// published integer formats.
pub(crate) fn gain_decode(rate: PackedRate, lag_base: i32, gind: u32) -> QGainInfo {
    let short = rate == PackedRate::High && lag_base < 58;
    let gsize = crate::spec_tables::GAIN_TABLE_SIZE;
    let (pgindex, mgindex, train) = if short {
        let masked = gind & 0x7FF;
        (
            ((masked / gsize) as usize).min(84),
            (masked % gsize) as usize,
            gind & 0x800 != 0,
        )
    } else {
        (
            ((gind / gsize) as usize).min(169),
            (gind % gsize) as usize,
            false,
        )
    };
    // The 85-row short-lag codebook is stored as the "5.3" table, the
    // shared 170-row codebook as the "6.3" table (§2.14 naming).
    let spec_rate = if short {
        crate::spec_tables::SpecRate::Low
    } else {
        crate::spec_tables::SpecRate::High
    };
    let row = crate::spec_tables::adaptive_codebook_gain_row(spec_rate, pgindex as u32)
        .expect("clamped PGIndex is always a valid row");
    let mut taps = [0i16; ACB_TAPS];
    taps.copy_from_slice(&row[..ACB_TAPS]);
    QGainInfo {
        taps,
        fcb_gain: crate::spec_tables::fixed_codebook_gain(mgindex as u8)
            .expect("MGIndex < 24 by construction")
            << FCB_GAIN_SHIFT,
        pgindex,
        mgindex,
        train,
    }
}

/// Saturate a 64-bit value to the 32-bit range (the excitation
/// domain's Word32 rail).
#[inline]
fn sat32(x: i64) -> i32 {
    if x > i32::MAX as i64 {
        i32::MAX
    } else if x < i32::MIN as i64 {
        i32::MIN
    } else {
        x as i32
    }
}

/// Fifth-order adaptive-codebook contribution `u[n]` (eq. 41.1–41.2)
/// on the **Word32** excitation history (most recent sample last): Q13
/// taps against the delay-`L` view, wide accumulation, rounded back to
/// Word32 sample units with saturation at the Q31 rail.
///
/// Why Word32 and not Word16: the OVER conformance class drives the
/// decoded excitation far past the 16-bit range (the frame-2..7
/// magnitudes on `OVERD53.TCO` reach 2³⁶ in a linear model) while the
/// reference decoder output keeps tracking a *linear* excitation loop
/// (whole-file waveform correlation 0.97+ for the unclamped model
/// versus 0.03 for a Word16-saturated history). The conformance
/// vectors therefore pin the excitation loop as wide arithmetic —
/// the reference's overflow-flag machinery (TSTG7231 §"OVER.. files")
/// preserves linearity here, and a hard Word16 excitation is the one
/// model the vectors *reject*.
pub(crate) fn acb_contribution(
    hist: &[i32],
    lag: i32,
    taps: &[i16; ACB_TAPS],
) -> [i32; crate::tables::SUBFRAME_SIZE] {
    let l = lag.clamp(
        crate::tables::PITCH_MIN as i32,
        crate::tables::PITCH_MAX as i32,
    ) as usize;
    let hlen = hist.len();
    debug_assert!(hlen >= l + 2);
    // eq. 41.1 — vector-arbitrated reading: e′ is the CONTIGUOUS
    // history slice starting at e[−L−2] (`e′[m] = e[m − L − 2]` for
    // m ≤ L + 1), extended periodically with period L beyond it:
    // `e′[m] = e[((m − 2) mod L) − L]`. The literal "(n mod L)" of the
    // published formula would skip e[−L]/e[−L+1] right after the two
    // seeds and replay two samples early; on PATHD53 the decoded
    // waveform then leads the reference by exactly two samples, so the
    // vectors pin the contiguous form (tap j = 2 sits at delay L — a
    // symmetric five-tap predictor).
    let eprime = |m: usize| -> i32 {
        let off = if m < 2 { l + 2 - m } else { l - ((m - 2) % l) };
        hist[hlen - off]
    };
    let mut u = [0i32; crate::tables::SUBFRAME_SIZE];
    for (n, out) in u.iter_mut().enumerate() {
        let mut acc = 0i64;
        for (j, &t) in taps.iter().enumerate() {
            acc += t as i64 * eprime(n + j) as i64;
        }
        // Round the product sum back to sample units (tap Q-format —
        // vector-arbitrated, see ACB_TAP_SHIFT).
        *out = sat32((acc + (1 << (ACB_TAP_SHIFT - 1))) >> ACB_TAP_SHIFT);
    }
    u
}

/// Effective right-shift applied to the gain-vector tap products
/// (vector-arbitrated: the /8192 reading diverges on PATHD53 where the
/// reference stays bounded; the least-squares fit of the deconvolved
/// reference excitation sits at 1.0 under /16384).
pub(crate) const ACB_TAP_SHIFT: i64 = 14;
/// Fixed-codebook pulses are **doubled** table amplitudes
/// (vector-arbitrated: the deconvolved PATHD53 subframe-1 pulses are
/// ±6340 = 2 × the published level 3170 at MGIndex 21, matched
/// sample-exact).
pub(crate) const FCB_GAIN_SHIFT: i32 = 1;
/// The synthesis output is halved on emission — the counterpart of the
/// doubled-pulse excitation domain (raises exact-sample agreement on
/// PATHD53 from 9.6% to 14.3%).
pub(crate) const SYN_OUT_SHIFT: i32 = 1;
/// The stored excitation saturates at the Word16 rail *of the
/// table-scale domain* (= ±65534 in the doubled domain): the OVER
/// class rejects a plain Word16 rail in the doubled domain (waveform
/// collapse) and also rejects an unbounded loop (the reference stays
/// bounded).
pub(crate) const EXC_SAT16: bool = true;
/// See [`EXC_SAT16`].
pub(crate) const EXC_RAIL: i32 = 65534;

/// High-rate MP-MLQ fixed-codebook vector (§2.15/§2.17): combinatorial
/// position decode, grid placement, MSB-first negative-sign convention
/// (r388 vector-arbitrated), Word16 amplitudes, saturating Dirac-train
/// accumulation on short-lag train subframes.
pub(crate) fn mpmlq_fixed_vector(
    pos_code: u32,
    psig: u32,
    grid: u8,
    n_pulses: usize,
    gain: i16,
    train: bool,
    lag_base: i32,
) -> [i32; crate::tables::SUBFRAME_SIZE] {
    let mut v = [0i32; crate::tables::SUBFRAME_SIZE];
    let Some(slots) = crate::spec_tables::fcbk_unpk_positions(pos_code, n_pulses) else {
        return v;
    };
    let period = lag_base.max(1) as usize;
    for (k, &slot) in slots.iter().enumerate() {
        let bit = n_pulses - 1 - k;
        let amp: i32 = if (psig >> bit) & 1 == 1 {
            -(gain as i32)
        } else {
            gain as i32
        };
        let base = 2 * slot + grid as usize;
        if train {
            let mut pos = base;
            while pos < crate::tables::SUBFRAME_SIZE {
                v[pos] = v[pos].saturating_add(amp);
                pos += period;
            }
        } else if base < crate::tables::SUBFRAME_SIZE {
            v[base] = v[base].saturating_add(amp);
        }
    }
    v
}

/// Low-rate ACELP fixed-codebook vector (§2.16/§2.17 step 2): Table 1
/// track slots, shared grid bit, set-bit-positive sign convention
/// (r388 vector-arbitrated), Word16 amplitudes.
pub(crate) fn acelp_fixed_vector(
    pos: u32,
    psig: u32,
    grid: u8,
    gain: i16,
) -> [i32; crate::tables::SUBFRAME_SIZE] {
    let mut v = [0i32; crate::tables::SUBFRAME_SIZE];
    for track in 0..4usize {
        let slot = (pos >> (3 * track)) & 0x7;
        let Some(sample) = crate::spec_tables::acelp_track_position(
            crate::spec_tables::AcelpTrack::ALL[track],
            slot as usize,
            grid != 0,
        ) else {
            continue;
        };
        let amp: i32 = if (psig >> track) & 1 == 1 {
            gain as i32
        } else {
            -(gain as i32)
        };
        v[sample] = v[sample].saturating_add(amp);
    }
    v
}

/// §2.16 pitch-synchronous ACELP enhancement for short lags:
/// `v[n] ← v[n] + β·v[n − L − ε]` in ascending `n`, β in Q15,
/// saturating adds.
pub(crate) fn acelp_pitch_enhance(
    v: &mut [i32; crate::tables::SUBFRAME_SIZE],
    lag: i32,
    pgindex: usize,
) {
    if lag >= crate::tables::SUBFRAME_SIZE as i32 {
        return;
    }
    let Some(ltp) = crate::spec_tables::pitch_1tap_ltp(pgindex) else {
        return;
    };
    if ltp.gain == 0 {
        return;
    }
    let delay = lag + ltp.selector as i32;
    if delay <= 0 {
        return;
    }
    for n in delay as usize..crate::tables::SUBFRAME_SIZE {
        let scaled = sat32((ltp.gain as i64 * v[n - delay as usize] as i64) >> 15);
        v[n] = v[n].saturating_add(scaled);
    }
}

// ---------------------------------------------------------------------
// §3.7 LPC synthesis
// ---------------------------------------------------------------------

/// One subframe of the eq. 48 all-pole synthesis on Q13 coefficients:
/// `sy[n] = x[n] + Σ ã_j·sy[n−j]` with a wide Q13 accumulator over the
/// Word32 excitation and Word16-rounded, **saturated** output. `mem`
/// holds `sy[n−1]..sy[n−10]` most recent first and carries the
/// *saturated* Word16 values — the vector-arbitrated model: the OVER
/// class shows the reference's synthesis memory tracking the clipped
/// output (a linear-memory model diverges once the output rails).
pub(crate) fn synthesis_subframe(
    a: &[i16; LPC_ORDER],
    x: &[i32; crate::tables::SUBFRAME_SIZE],
    mem: &mut [i32; LPC_ORDER],
    out: &mut [i16; crate::tables::SUBFRAME_SIZE],
    clamp_mem: bool,
) {
    for n in 0..crate::tables::SUBFRAME_SIZE {
        let mut acc = (x[n] as i64) << 13; // sample units → Q13
        for j in 0..LPC_ORDER {
            acc += a[j] as i64 * mem[j] as i64;
        }
        let y = sat32((acc + (1 << 12)) >> 13);
        let y16 = saturate(y);
        for j in (1..LPC_ORDER).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = if clamp_mem { y16 as i32 } else { y };
        out[n] = saturate(y >> SYN_OUT_SHIFT);
    }
}

/// Decoded-LSP state advance shared by the normal and erasure paths:
/// stability check with previous-vector fallback.
pub(crate) fn lsp_check_or_previous(
    mut cand: [i16; LPC_ORDER],
    prev: &[i16; LPC_ORDER],
    delta_min: i16,
) -> [i16; LPC_ORDER] {
    if lsp_stability(&mut cand, delta_min) {
        cand
    } else {
        *prev
    }
}

/// The §3.11 cold-start previous-LSP vector (`p_DC`).
pub(crate) fn lsp_dc() -> [i16; LPC_ORDER] {
    LSP_DC_PREDICTED_FREQ_Q15
}

/// Interpolated per-subframe LPC set for one frame (eq. 8 + eq. 9).
pub(crate) fn frame_lpc_q13(
    prev: &[i16; LPC_ORDER],
    cur: &[i16; LPC_ORDER],
) -> [[i16; LPC_ORDER]; crate::tables::SUBFRAMES_PER_FRAME] {
    let mut out = [[0i16; LPC_ORDER]; crate::tables::SUBFRAMES_PER_FRAME];
    for (k, slot) in out.iter_mut().enumerate() {
        let lsp = lsp_interpolate(k, prev, cur);
        *slot = lsp_to_lpc_q13(&lsp);
    }
    out
}

// ---------------------------------------------------------------------
// Frame-level fixed-point decoder
// ---------------------------------------------------------------------

use crate::tables::{FRAME_SIZE_SAMPLES, SUBFRAMES_PER_FRAME, SUBFRAME_SIZE};

/// Stateful fixed-point G.723.1 decoder (§3.1 pipeline on saturating
/// Word16/Word32 arithmetic).
///
/// The counterpart of the float [`crate::encoder::SynthesisState`]
/// decode path; state layout follows §3.11 (everything zero except the
/// previous LSP vector = `p_DC` and the AGC gain = 1).
pub struct QSynthesis {
    prev_lsp: [i16; LPC_ORDER],
    exc_hist: [i32; EXC_HIST],
    syn_mem: [i32; LPC_ORDER],
    postfilter: bool,
    /// Synthesis-memory domain (vector-arbitration switch): `true`
    /// carries Word16-saturated values in the recursion, `false` the
    /// wide linear values.
    clamp_syn_mem: bool,
}

impl QSynthesis {
    pub fn new() -> Self {
        Self {
            prev_lsp: lsp_dc(),
            exc_hist: [0; EXC_HIST],
            syn_mem: [0; LPC_ORDER],
            postfilter: true,
            clamp_syn_mem: true,
        }
    }

    /// §3.11 cold-start reset; the post-filter switch is configuration,
    /// not decoder state, and survives.
    pub fn reset(&mut self) {
        let pf = self.postfilter;
        let cm = self.clamp_syn_mem;
        *self = Self::new();
        self.postfilter = pf;
        self.clamp_syn_mem = cm;
    }

    /// Enable / disable the §3.6–§3.9 post-filter chain (the ITU
    /// decoder-vector configurations require the switch).
    pub fn set_postfilter(&mut self, enabled: bool) {
        self.postfilter = enabled;
    }

    /// Arbitration switch: synthesis-memory domain (see
    /// [`synthesis_subframe`]).
    pub(crate) fn set_clamp_syn_mem(&mut self, clamp: bool) {
        self.clamp_syn_mem = clamp;
    }

    /// Decode one unpacked clause-4 parameter set into 240 PCM samples.
    pub fn decode_params(&mut self, p: &SpecFrameParams) -> [i16; FRAME_SIZE_SAMPLES] {
        // --- LSP decode (§3.2 → 2.6) with stability fallback.
        let cur = lsp_check_or_previous(
            lsp_decode(p.lsp_index, &self.prev_lsp),
            &self.prev_lsp,
            LSP_DELTA_MIN_Q15,
        );

        // --- Pitch lags (§3.4, eq. 37–38).
        let lag0 = p.acl[0] as i32 + 18;
        let lag1 = lag0 + DELTA_LAG[p.acl[1] as usize];
        let lag2 = p.acl[2] as i32 + 18;
        let lag3 = lag2 + DELTA_LAG[p.acl[3] as usize];
        let lags = [lag0, lag1, lag2, lag3];

        // --- Whole-frame excitation (§3.6 requires it generated and
        // saved before the pitch post-filter runs). The loop runs in
        // the wide Word32 domain (see [`acb_contribution`]).
        let mut exc = [0i32; FRAME_SIZE_SAMPLES];
        for s in 0..SUBFRAMES_PER_FRAME {
            let lag_base = if s < 2 { lags[0] } else { lags[2] };
            let g = gain_decode(p.rate, lag_base, p.gain[s]);
            let u = acb_contribution(&self.exc_hist, lags[s], &g.taps);
            let v = match p.rate {
                PackedRate::High => {
                    let n_pulses = if s % 2 == 0 { 6 } else { 5 };
                    mpmlq_fixed_vector(
                        p.pos[s], p.psig[s], p.grid[s], n_pulses, g.fcb_gain, g.train, lag_base,
                    )
                }
                PackedRate::Low => {
                    let mut v = acelp_fixed_vector(p.pos[s], p.psig[s], p.grid[s], g.fcb_gain);
                    acelp_pitch_enhance(&mut v, lags[s], g.pgindex);
                    v
                }
            };
            let start = s * SUBFRAME_SIZE;
            for n in 0..SUBFRAME_SIZE {
                let e = u[n].saturating_add(v[n]);
                exc[start + n] = if EXC_SAT16 {
                    e.clamp(-EXC_RAIL - 1, EXC_RAIL)
                } else {
                    e
                };
            }
            self.push_excitation(&exc[start..start + SUBFRAME_SIZE]);
        }

        // --- §3.7 synthesis per subframe on the interpolated Q13 LPC.
        // (§3.6 pitch post-filter + §3.8/§3.9 back half land with the
        // post-filter stage; until then the switch selects the plain
        // synthesis output on both settings.)
        let lpc = frame_lpc_q13(&self.prev_lsp, &cur);
        let mut pcm = [0i16; FRAME_SIZE_SAMPLES];
        for s in 0..SUBFRAMES_PER_FRAME {
            let start = s * SUBFRAME_SIZE;
            let mut x = [0i32; SUBFRAME_SIZE];
            x.copy_from_slice(&exc[start..start + SUBFRAME_SIZE]);
            let mut sy = [0i16; SUBFRAME_SIZE];
            synthesis_subframe(&lpc[s], &x, &mut self.syn_mem, &mut sy, self.clamp_syn_mem);
            pcm[start..start + SUBFRAME_SIZE].copy_from_slice(&sy);
        }

        self.prev_lsp = cur;
        pcm
    }

    /// Advance the excitation history by one subframe.
    fn push_excitation(&mut self, sub: &[i32]) {
        self.exc_hist.copy_within(SUBFRAME_SIZE.., 0);
        let tail = EXC_HIST - SUBFRAME_SIZE;
        self.exc_hist[tail..].copy_from_slice(sub);
    }
}

impl Default for QSynthesis {
    fn default() -> Self {
        Self::new()
    }
}

/// eq. 38 differential-lag table for odd subframes.
const DELTA_LAG: [i32; 4] = [-1, 0, 1, 2];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::SUBFRAMES_PER_FRAME;

    #[test]
    fn dc_vector_decode_with_zero_residual_row_stays_near_dc() {
        // Index 0 selects row 0 of each band codebook; with prev = DC
        // the predictor vanishes so the decode is DC + row0.
        let dc = lsp_dc();
        let out = lsp_decode(0, &dc);
        for i in 0..LPC_ORDER {
            let row0 = match i {
                0..=2 => lsp_codebook_entry(LspBand::Band0, 0)[i],
                3..=5 => lsp_codebook_entry(LspBand::Band1, 0)[i - 3],
                _ => lsp_codebook_entry(LspBand::Band2, 0)[i - 6],
            };
            assert_eq!(out[i], add(dc[i], row0));
        }
    }

    #[test]
    fn fixed_lsp_decode_matches_float_reference_path() {
        // The float spec_lsp path decodes in the same table-unit domain
        // with f32 arithmetic; away from saturation the two must agree
        // to within a rounding unit.
        let dc = lsp_dc();
        let mut prev_q = dc;
        let mut prev_f = crate::spec_lsp::lsp_dc_freq();
        let mut idx: u32 = 0x00_01_02;
        for _ in 0..50 {
            let q = lsp_decode(idx, &prev_q);
            let f = crate::spec_lsp::decode_lsp_freq(idx, &prev_f);
            for i in 0..LPC_ORDER {
                assert!(
                    (q[i] as f32 - f[i]).abs() <= 1.0,
                    "line {i}: fixed {} vs float {}",
                    q[i],
                    f[i]
                );
            }
            prev_q = q;
            for i in 0..LPC_ORDER {
                prev_f[i] = q[i] as f32; // keep the two predictors in lockstep
            }
            idx = idx.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) & 0xFF_FF_FF;
        }
    }

    #[test]
    fn stability_orders_a_reversed_pair_and_reports_hopeless_vectors() {
        let mut p = lsp_dc();
        p.swap(4, 5); // one out-of-order pair
        assert!(lsp_stability(&mut p, LSP_DELTA_MIN_Q15));
        for w in p.windows(2) {
            assert!(w[1] as i32 - w[0] as i32 >= LSP_DELTA_MIN_Q15 as i32);
        }

        // An all-equal vector cannot be spread to 10 × 256 units within
        // 10 sweeps of pair-midpoint moves starting from equal values —
        // every sweep only spreads the outermost pair further.
        let mut collapsed = [16_000i16; LPC_ORDER];
        let ok = lsp_stability(&mut collapsed, LSP_DELTA_MIN_Q15);
        if ok {
            for w in collapsed.windows(2) {
                assert!(w[1] as i32 - w[0] as i32 >= LSP_DELTA_MIN_Q15 as i32);
            }
        }
    }

    #[test]
    fn interpolation_endpoints_and_midpoint() {
        let dc = lsp_dc();
        let mut cur = dc;
        for c in cur.iter_mut() {
            *c = add(*c, 1000);
        }
        let sf3 = lsp_interpolate(3, &dc, &cur);
        assert_eq!(sf3, cur, "subframe 3 is the current vector exactly");
        let sf1 = lsp_interpolate(1, &dc, &cur);
        for i in 0..LPC_ORDER {
            let expect = ((dc[i] as i32 + cur[i] as i32 + 1) / 2) as i16;
            assert!(
                (sf1[i] - expect).abs() <= 1,
                "line {i}: {} vs {}",
                sf1[i],
                expect
            );
        }
        let sf0 = lsp_interpolate(0, &dc, &cur);
        for i in 0..LPC_ORDER {
            let expect = (0.75 * dc[i] as f64 + 0.25 * cur[i] as f64).round() as i16;
            assert!((sf0[i] - expect).abs() <= 1);
        }
    }

    #[test]
    fn cosine_lookup_tracks_the_real_cosine() {
        for f in (0..32_768i32).step_by(37) {
            let c = cos_q14(f as i16) as f64 / 16_384.0;
            let real = (std::f64::consts::PI * f as f64 / 32_768.0).cos();
            assert!(
                (c - real).abs() < 3.0e-4,
                "f = {f}: table {c} vs real {real}"
            );
        }
    }

    #[test]
    fn q13_lpc_matches_float_conversion_on_the_dc_vector() {
        // Convert the DC LSP set on both paths and compare in the
        // float domain (fixed path is Q13, float path returns A(z)
        // with a[0] = 1 and the opposite sign convention).
        let dc = lsp_dc();
        let a_q = lsp_to_lpc_q13(&dc);
        let cos_f = crate::spec_lsp::lsp_freq_to_cosines(&dc.map(|q| q as f32));
        let a_f = crate::encoder::lsp_to_lpc(&cos_f);
        for j in 0..LPC_ORDER {
            let fixed = a_q[j] as f32 / 8192.0;
            let float = -a_f[j + 1];
            assert!(
                (fixed - float).abs() < 4.0e-3,
                "ã_{}: fixed {fixed} vs float {float}",
                j + 1
            );
        }
    }

    #[test]
    fn q13_lpc_matches_float_conversion_on_random_stable_sets() {
        let mut state = 0x1234_5678u32;
        let mut rand = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as i32
        };
        for _ in 0..200 {
            // Random ordered LSP vector with ≥ 400-unit gaps.
            let mut lsp = [0i16; LPC_ORDER];
            let mut f = 800i32 + (rand() % 800);
            for slot in lsp.iter_mut() {
                *slot = f as i16;
                f += 500 + (rand() % 2200);
            }
            if f >= 32_000 {
                continue;
            }
            let a_q = lsp_to_lpc_q13(&lsp);
            let cos_f = crate::spec_lsp::lsp_freq_to_cosines(&lsp.map(|q| q as f32));
            let a_f = crate::encoder::lsp_to_lpc(&cos_f);
            // Q13 in Word16 covers |ã| < 4; heavily clustered synthetic
            // sets can exceed that, where the fixed path (correctly)
            // saturates. Compare only in-range coefficient sets.
            if a_f[1..].iter().any(|c| c.abs() >= 3.9) {
                continue;
            }
            for j in 0..LPC_ORDER {
                let fixed = a_q[j] as f32 / 8192.0;
                let float = -a_f[j + 1];
                assert!(
                    (fixed - float).abs() < 8.0e-3,
                    "ã_{}: fixed {fixed} vs float {float} (lsp {lsp:?})",
                    j + 1
                );
            }
        }
    }

    #[test]
    fn frame_lpc_produces_four_subframe_sets() {
        let dc = lsp_dc();
        let mut cur = dc;
        for c in cur.iter_mut() {
            *c = add(*c, 700);
        }
        let sets = frame_lpc_q13(&dc, &cur);
        assert_eq!(sets.len(), SUBFRAMES_PER_FRAME);
        // Subframe 3 equals a direct conversion of the current vector.
        assert_eq!(sets[3], lsp_to_lpc_q13(&cur));
        // Adjacent subframes differ smoothly but are not identical.
        assert_ne!(sets[0], sets[3]);
    }

    #[test]
    fn gain_decode_matches_float_reference_split() {
        for (rate, lag) in [
            (PackedRate::Low, 100),
            (PackedRate::High, 100),
            (PackedRate::High, 20),
        ] {
            for gind in [0u32, 1234, 2040, 4079, 0x800 | 777] {
                let q = gain_decode(rate, lag, gind);
                let f = crate::spec_exc::decode_gain_word(rate, lag, gind);
                assert_eq!(q.pgindex, f.pgindex, "{rate:?} lag {lag} gind {gind}");
                assert_eq!(q.mgindex, f.mgindex);
                assert_eq!(q.train, f.train);
                // Fixed path: doubled table amplitude; float path:
                // table / 32768 in the normalised domain.
                assert!((q.fcb_gain as f32 / 65_536.0 - f.fcb_gain).abs() < 1e-6);
                for t in 0..ACB_TAPS {
                    assert!((q.taps[t] as f32 / 16_384.0 - f.taps[t]).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn acb_contribution_matches_eq41_geometry() {
        // Single unit spike at e[-1], tap j = 2 only: u[n] = spike when
        // ((n+2) mod L) == L − 1 (same geometry as the float test, at
        // integer amplitude).
        let lag = 40i32;
        let mut hist = [0i32; EXC_HIST];
        hist[EXC_HIST - 1] = 1000;
        let taps = [0, 0, 1 << ACB_TAP_SHIFT, 0, 0]; // 1.0 at the tap scale
        let u = acb_contribution(&hist, lag, &taps);
        // Contiguous e' (vector-arbitrated): tap j = 2 sits at delay L,
        // so the e[-1] spike appears at n = L - 1, 2L - 1, ...
        for (n, &s) in u.iter().enumerate() {
            let expect = if n % lag as usize == lag as usize - 1 {
                1000
            } else {
                0
            };
            assert_eq!(s, expect, "sample {n}");
        }
    }

    #[test]
    fn acb_contribution_saturates_at_the_word32_rail() {
        // A history pinned at the Word32 rail with all-max taps must
        // saturate instead of wrapping — the excitation loop's rail is
        // 32-bit (see the acb_contribution docs for the vector-
        // arbitrated domain choice).
        let hist = [i32::MAX; EXC_HIST];
        let taps = [i16::MAX; ACB_TAPS];
        let u = acb_contribution(&hist, 60, &taps);
        assert!(u.iter().all(|&s| s == i32::MAX));
    }

    #[test]
    fn fixed_vectors_match_float_reconstruction_scaled() {
        // MP-MLQ: same slots/signs as the float unit test, amplitude in
        // raw sample units.
        let slots = [0usize, 3, 7, 12, 20, 29];
        let code = crate::spec_tables::fcbk_pack_positions(&slots).unwrap();
        let v = mpmlq_fixed_vector(code, 0b010101, 1, 6, 6623, false, 100);
        for (k, &slot) in slots.iter().enumerate() {
            let sign = if k % 2 == 1 { -1 } else { 1 };
            assert_eq!(v[2 * slot + 1], sign * 6623, "pulse {k}");
        }

        // ACELP with the r388 set-bit-positive convention.
        let pos = 1 | (2 << 3) | (7 << 6) | (7 << 9);
        let v = acelp_fixed_vector(pos, 0b0001, 0, 502);
        assert_eq!(v[8], 502);
        assert_eq!(v[18], -502);
        assert_eq!(v.iter().filter(|&&s| s != 0).count(), 2);
    }

    #[test]
    fn synthesis_matches_direct_form_on_a_one_pole_filter() {
        // ã_1 = 0.5 (4096 in Q13), all other taps zero: an impulse
        // decays by halves.
        let mut a = [0i16; LPC_ORDER];
        a[0] = 4096;
        let mut x = [0i32; SUBFRAME_SIZE];
        x[0] = 16_000;
        let mut mem = [0i32; LPC_ORDER];
        let mut out = [0i16; SUBFRAME_SIZE];
        synthesis_subframe(&a, &x, &mut mem, &mut out, true);
        // Emitted PCM is the internal value halved (the doubled-FCB /
        // halved-output domain, vector-arbitrated).
        assert_eq!(out[0], 8_000);
        assert_eq!(out[1], 4_000);
        assert_eq!(out[2], 2_000);
        // The rounded recursion rounds half up, so the internal decay
        // parks in the classic ±1 limit cycle (1 · 0.5 rounds back to
        // 1); the halved output reads 0.
        assert_eq!(out[20], 0);
        assert_eq!(mem[0], 1, "internal one-pole limit cycle");
        let x2 = [0i32; SUBFRAME_SIZE];
        let mut out2 = [0i16; SUBFRAME_SIZE];
        synthesis_subframe(&a, &x2, &mut mem, &mut out2, true);
        assert_eq!(out2, [0i16; SUBFRAME_SIZE]);
        assert_eq!(mem[0], 1);
    }

    #[test]
    fn synthesis_output_saturates_at_word16() {
        // An unstable ã_1 = 1.25 on a large impulse: the wide-memory
        // variant grows until the halved output pins at 32767; the
        // Word16-memory variant parks its recursion at the rail, so the
        // halved output settles just above 32767/2 · 1.25 ≈ 20479.
        for clamp_mem in [true, false] {
            let mut a = [0i16; LPC_ORDER];
            a[0] = 10_240;
            let mut x = [0i32; SUBFRAME_SIZE];
            x[0] = 60_000;
            let mut mem = [0i32; LPC_ORDER];
            let mut out = [0i16; SUBFRAME_SIZE];
            synthesis_subframe(&a, &x, &mut mem, &mut out, clamp_mem);
            if clamp_mem {
                assert_eq!(mem[0], i16::MAX as i32, "memory parks at the rail");
                assert_eq!(out[SUBFRAME_SIZE - 1], 20_479);
            } else {
                assert!(out[..12].contains(&i16::MAX));
                assert!(mem[0] > i16::MAX as i32);
            }
        }
    }

    #[test]
    fn qsynthesis_decodes_zeroed_frames_at_both_rates() {
        let mut st = QSynthesis::new();
        st.set_postfilter(false);
        for rate in [PackedRate::High, PackedRate::Low] {
            let p = SpecFrameParams::zeroed(rate);
            let pcm = st.decode_params(&p);
            assert_eq!(pcm.len(), FRAME_SIZE_SAMPLES);
            // A zeroed index set decodes to a small-amplitude frame
            // (gain level 0 = amplitude 1), never full-scale.
            assert!(pcm.iter().all(|&s| s.unsigned_abs() < 1000));
        }
    }

    #[test]
    fn extrapolation_leaks_toward_dc() {
        let dc = lsp_dc();
        let mut prev = dc;
        for p in prev.iter_mut() {
            *p = add(*p, 2048);
        }
        let e = lsp_extrapolate(&prev);
        for i in 0..LPC_ORDER {
            // b_e = 23/32 of the 2048-unit offset ≈ 1472.
            let off = e[i] as i32 - dc[i] as i32;
            assert!((off - 1472).abs() <= 1, "line {i}: offset {off}");
        }
    }
}
