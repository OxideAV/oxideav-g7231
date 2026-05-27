//! ITU-T G.723.1 bit-exact numeric tables, vendored from
//! `docs/audio/g7231/tables/` per the workspace's clean-room policy.
//!
//! These are *facts* (numeric constants the codec design depends on)
//! rather than algorithmic expression — they sit on the
//! *Feist v. Rural Telephone Service Co.* (1991) side of the
//! copyright line. The workspace's extractor reads only the
//! data-only table listing of the ITU reference fixed-point source
//! distribution and writes one CSV per table, with a `.meta` sidecar
//! recording the upstream SHA-256, source line range, C identifier,
//! and resolved dimensions. The CSVs vendored into `tables/` of this
//! crate are byte-identical to the workspace copies; see
//! `tables/README.md` for the per-table inventory + provenance.
//!
//! This module exposes the smaller foundational tables (LPC analysis
//! primitives, LSP DC predictor, perceptual / postfilter coefficients,
//! highpass-filter constants, gain-quantiser decision factors,
//! bit-allocation segment table, and the rate-6.3 MP-MLQ
//! pulse-count / max-position tables). Larger codebooks (the
//! 768/768/1024-entry LSP split VQ, the 1 700 + 3 400-entry
//! adaptive-codebook gain tables, the MP-MLQ combinatorial table,
//! the 1-tap LTP shortcut tables, and the per-rate taming-gain
//! tables) remain in the workspace area for now and will be wired
//! into this crate in follow-up rounds once their consumers exist.
//!
//! # Q format
//!
//! Tables tagged `Q15` in their filename are fixed-point fractional
//! values in the range `[-1.0, 1.0)` scaled by `2^15`. Convert to
//! `f32` with `(value as f32) / 32_768.0` when a floating-point
//! consumer wants the underlying real number.
//!
//! # API shape
//!
//! Each table is exposed as a `pub fn name() -> &'static [T; N]`
//! accessor that lazily parses the embedded CSV on first call and
//! caches the result in a static `OnceLock`. The accessor pattern
//! (rather than `static FOO: [T; N] = …`) lets us keep the data
//! in the CSV source-of-truth without re-typing values into Rust
//! literals.

use std::sync::OnceLock;

// ---------- embedded CSV bodies ----------

const HAMMING_WINDOW_CSV: &str = include_str!("../tables/lpc-hamming-window-Q15.csv");
const BINOMIAL_LAG_WINDOW_CSV: &str = include_str!("../tables/lpc-binomial-lag-window-Q15.csv");
const BANDWIDTH_EXPANSION_CSV: &str = include_str!("../tables/lpc-bandwidth-expansion-Q15.csv");
const LSP_COSINE_LOOKUP_CSV: &str = include_str!("../tables/lpc-lsp-cosine-lookup-Q15.csv");
const LSP_DC_PREDICTED_CSV: &str = include_str!("../tables/lsp-dc-predicted-frequencies-Q15.csv");
const LSP_BAND_INFO_CSV: &str = include_str!("../tables/lsp-band-info.csv");
const PERCEPTUAL_ZERO_CSV: &str =
    include_str!("../tables/perceptual-weighting-zero-coefficients-Q15.csv");
const PERCEPTUAL_POLE_CSV: &str =
    include_str!("../tables/perceptual-weighting-pole-coefficients-Q15.csv");
const POSTFILTER_ZERO_CSV: &str = include_str!("../tables/postfilter-zero-coefficients-Q15.csv");
const POSTFILTER_POLE_CSV: &str = include_str!("../tables/postfilter-pole-coefficients-Q15.csv");
const HIGHPASS_CONSTANTS_CSV: &str = include_str!("../tables/highpass-filter-constants-Q15.csv");
const GAIN_QUANT_DECISION_FACTORS_CSV: &str =
    include_str!("../tables/gain-quantizer-decision-factors.csv");
const BIT_ALLOC_BOUNDARIES_CSV: &str =
    include_str!("../tables/bit-allocation-segment-boundaries.csv");
const BIT_ALLOC_BASE_CSV: &str = include_str!("../tables/bit-allocation-segment-base.csv");
const MP_MLQ_PULSE_COUNT_CSV: &str = include_str!("../tables/mp-mlq-pulse-count-per-subframe.csv");
const MP_MLQ_MAX_POSITION_CSV: &str = include_str!("../tables/mp-mlq-max-position-table.csv");

// ---------- generic CSV value parser ----------

/// Parse one numeric value: either a `0x`-prefixed hex literal (parsed
/// as unsigned then sign-extended into i64 at whatever the caller's
/// width turns out to be) or a signed decimal integer.
fn parse_value(field: &str, ctx: &str) -> i64 {
    let trimmed = field.trim();
    if let Some(rest) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(rest, 16)
            .unwrap_or_else(|_| panic!("{ctx}: invalid hex literal {trimmed:?}")) as i64
    } else {
        trimmed
            .parse::<i64>()
            .unwrap_or_else(|_| panic!("{ctx}: invalid integer literal {trimmed:?}"))
    }
}

fn as_i16(v: i64, ctx: &str) -> i16 {
    // Hex literals like 0xc000 should reinterpret as a negative i16
    // (Word16 in C is the 16-bit two's-complement type). Decimal
    // values are signed already.
    if (0..=0xFFFF).contains(&v) {
        v as u16 as i16
    } else if (i16::MIN as i64..=i16::MAX as i64).contains(&v) {
        v as i16
    } else {
        panic!("{ctx}: value {v} does not fit in i16");
    }
}

fn as_i32(v: i64, ctx: &str) -> i32 {
    if (0..=0xFFFF_FFFF).contains(&v) {
        v as u32 as i32
    } else if (i32::MIN as i64..=i32::MAX as i64).contains(&v) {
        v as i32
    } else {
        panic!("{ctx}: value {v} does not fit in i32");
    }
}

/// Parse a one-value-per-line CSV into an `[i16; N]`. Panics if the
/// CSV does not yield exactly `N` parsable values.
fn parse_i16_array<const N: usize>(csv: &str, name: &str) -> [i16; N] {
    let mut out = [0i16; N];
    let mut i = 0usize;
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        assert!(i < N, "{name}: too many entries (>{N})");
        out[i] = as_i16(parse_value(line, name), name);
        i += 1;
    }
    assert!(i == N, "{name}: expected {N} entries, got {i}");
    out
}

/// Parse a one-value-per-line CSV into an `[i32; N]`.
fn parse_i32_array<const N: usize>(csv: &str, name: &str) -> [i32; N] {
    let mut out = [0i32; N];
    let mut i = 0usize;
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        assert!(i < N, "{name}: too many entries (>{N})");
        out[i] = as_i32(parse_value(line, name), name);
        i += 1;
    }
    assert!(i == N, "{name}: expected {N} entries, got {i}");
    out
}

/// Parse a `start,length`-per-line CSV into a `[[i16; 2]; N]`.
fn parse_pair_i16_array<const N: usize>(csv: &str, name: &str) -> [[i16; 2]; N] {
    let mut out = [[0i16; 2]; N];
    let mut i = 0usize;
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        assert!(i < N, "{name}: too many rows (>{N})");
        let mut it = line.split(',');
        let a = it
            .next()
            .unwrap_or_else(|| panic!("{name}: row {i} missing first field"));
        let b = it
            .next()
            .unwrap_or_else(|| panic!("{name}: row {i} missing second field"));
        assert!(
            it.next().is_none(),
            "{name}: row {i} has more than 2 fields"
        );
        out[i][0] = as_i16(parse_value(a, name), name);
        out[i][1] = as_i16(parse_value(b, name), name);
        i += 1;
    }
    assert!(i == N, "{name}: expected {N} rows, got {i}");
    out
}

// ---------- per-table accessors ----------

/// LPC Hamming window applied to the 180-sample analysis frame, §2.4.
///
/// `HammingWindowTable[180]`, Word16 / Q15.
pub fn hamming_window() -> &'static [i16; 180] {
    static T: OnceLock<[i16; 180]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(HAMMING_WINDOW_CSV, "lpc-hamming-window-Q15.csv"))
}

/// Binomial (white-noise correction) lag window applied to the
/// autocorrelation values during LPC analysis, §2.4.
///
/// `BinomialWindowTable[10]`, Word16 / Q15.
pub fn binomial_lag_window() -> &'static [i16; 10] {
    static T: OnceLock<[i16; 10]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(BINOMIAL_LAG_WINDOW_CSV, "lpc-binomial-lag-window-Q15.csv"))
}

/// Bandwidth-expansion factors `gamma^i` for the unquantised LPC
/// coefficients, §2.4.
///
/// `BandExpTable[10]`, Word16 / Q15.
pub fn bandwidth_expansion() -> &'static [i16; 10] {
    static T: OnceLock<[i16; 10]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(BANDWIDTH_EXPANSION_CSV, "lpc-bandwidth-expansion-Q15.csv"))
}

/// LSP root-finding cosine lookup table, §2.4.
///
/// `CosineTable[512]`, Word16. One full period of `16384 · cos(2πk/512)` —
/// effectively a Q14-scaled cosine (peak amplitude `16384`, not the
/// full Q15 `32767`). The Q14 scale leaves one extra headroom bit
/// for the Q15 multiplies the LSP root-bisection accumulator does
/// downstream. The filename retains the `-Q15` suffix used in the
/// workspace's CSV inventory for naming consistency across all
/// fractional-coefficient tables; the per-element scale is Q14.
pub fn lsp_cosine_lookup() -> &'static [i16; 512] {
    static T: OnceLock<[i16; 512]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(LSP_COSINE_LOOKUP_CSV, "lpc-lsp-cosine-lookup-Q15.csv"))
}

/// LSP DC-predicted frequencies — the reference vector subtracted
/// before split-VQ encoding, §2.6.
///
/// `LspDcTable[10]`, Word16 / Q15.
pub fn lsp_dc_predicted_frequencies() -> &'static [i16; 10] {
    static T: OnceLock<[i16; 10]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(LSP_DC_PREDICTED_CSV, "lsp-dc-predicted-frequencies-Q15.csv"))
}

/// LSP split-VQ band partition info: `(start, length)` per band, §2.6.
///
/// `BandInfoTable[3][2]`, Word16. Bands cover LSP indices
/// `[start..start+length)`; the three bands together cover the
/// 10-coefficient LSP vector.
pub fn lsp_band_info() -> &'static [[i16; 2]; 3] {
    static T: OnceLock<[[i16; 2]; 3]> = OnceLock::new();
    T.get_or_init(|| parse_pair_i16_array(LSP_BAND_INFO_CSV, "lsp-band-info.csv"))
}

/// Perceptual-weighting filter numerator (zero) coefficients, §2.9.
///
/// `PerFiltZeroTable[10]`, Word16 / Q15.
pub fn perceptual_weighting_zero() -> &'static [i16; 10] {
    static T: OnceLock<[i16; 10]> = OnceLock::new();
    T.get_or_init(|| {
        parse_i16_array(
            PERCEPTUAL_ZERO_CSV,
            "perceptual-weighting-zero-coefficients-Q15.csv",
        )
    })
}

/// Perceptual-weighting filter denominator (pole) coefficients, §2.9.
///
/// `PerFiltPoleTable[10]`, Word16 / Q15.
pub fn perceptual_weighting_pole() -> &'static [i16; 10] {
    static T: OnceLock<[i16; 10]> = OnceLock::new();
    T.get_or_init(|| {
        parse_i16_array(
            PERCEPTUAL_POLE_CSV,
            "perceptual-weighting-pole-coefficients-Q15.csv",
        )
    })
}

/// Post-filter numerator (zero) coefficients, §2.9.
///
/// `PostFiltZeroTable[10]`, Word16 / Q15.
pub fn postfilter_zero() -> &'static [i16; 10] {
    static T: OnceLock<[i16; 10]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(POSTFILTER_ZERO_CSV, "postfilter-zero-coefficients-Q15.csv"))
}

/// Post-filter denominator (pole) coefficients, §2.9.
///
/// `PostFiltPoleTable[10]`, Word16 / Q15.
pub fn postfilter_pole() -> &'static [i16; 10] {
    static T: OnceLock<[i16; 10]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(POSTFILTER_POLE_CSV, "postfilter-pole-coefficients-Q15.csv"))
}

/// Input-signal preprocessing low-pass / high-pass filter constants, §2.2.
///
/// `LpfConstTable[2]`, Word16 / Q15.
pub fn highpass_filter_constants() -> &'static [i16; 2] {
    static T: OnceLock<[i16; 2]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(HIGHPASS_CONSTANTS_CSV, "highpass-filter-constants-Q15.csv"))
}

/// Gain-quantiser decision factors, §2.14.
///
/// `fact[4]`, Word16. Values: `{273, 998, 499, 333}`.
pub fn gain_quantizer_decision_factors() -> &'static [i16; 4] {
    static T: OnceLock<[i16; 4]> = OnceLock::new();
    T.get_or_init(|| {
        parse_i16_array(
            GAIN_QUANT_DECISION_FACTORS_CSV,
            "gain-quantizer-decision-factors.csv",
        )
    })
}

/// Bit-allocation segment boundaries used by the bit-packer.
///
/// `L_bseg[3]`, Word32. Values: `{2048, 18432, 231233}`.
pub fn bit_allocation_segment_boundaries() -> &'static [i32; 3] {
    static T: OnceLock<[i32; 3]> = OnceLock::new();
    T.get_or_init(|| {
        parse_i32_array(
            BIT_ALLOC_BOUNDARIES_CSV,
            "bit-allocation-segment-boundaries.csv",
        )
    })
}

/// Bit-allocation segment base offsets (companion to `L_bseg`).
///
/// `base[3]`, Word16. Values: `{0, 32, 96}`.
pub fn bit_allocation_segment_base() -> &'static [i16; 3] {
    static T: OnceLock<[i16; 3]> = OnceLock::new();
    T.get_or_init(|| parse_i16_array(BIT_ALLOC_BASE_CSV, "bit-allocation-segment-base.csv"))
}

/// MP-MLQ pulse count per subframe (6.3 kbit/s mode), §2.13.
///
/// `Nb_puls[4]`, Word16. Values: `{6, 5, 6, 5}` — six pulses on
/// subframes 0 and 2, five on subframes 1 and 3.
pub fn mp_mlq_pulse_count_per_subframe() -> &'static [i16; 4] {
    static T: OnceLock<[i16; 4]> = OnceLock::new();
    T.get_or_init(|| {
        parse_i16_array(
            MP_MLQ_PULSE_COUNT_CSV,
            "mp-mlq-pulse-count-per-subframe.csv",
        )
    })
}

/// MP-MLQ max-position table for the codeword search, §2.13.
///
/// `MaxPosTable[4]`, Word32. Used to bound the candidate-position
/// enumeration in the pulse search.
pub fn mp_mlq_max_position_table() -> &'static [i32; 4] {
    static T: OnceLock<[i32; 4]> = OnceLock::new();
    T.get_or_init(|| parse_i32_array(MP_MLQ_MAX_POSITION_CSV, "mp-mlq-max-position-table.csv"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_window_loads_180_values() {
        let w = hamming_window();
        assert_eq!(w.len(), 180);
        // First value: 2621 ≈ 0.08 · 2^15 — a Hamming window at index 0
        // (0.54 − 0.46·cos(0) = 0.08).
        assert_eq!(w[0], 2_621);
        // Window is monotonically rising into the centre.
        assert!(w[0] < w[10]);
        assert!(w[10] < w[50]);
        // No value out of Q15 non-negative range (Hamming is always
        // non-negative: weights ∈ [0.08, 1.0]).
        for &v in w.iter() {
            assert!((0..=i16::MAX).contains(&v));
        }
    }

    #[test]
    fn binomial_lag_window_loads_10_values() {
        let w = binomial_lag_window();
        assert_eq!(w.len(), 10);
        // First entry should be just under 1.0 in Q15 (32 749 < 32 768).
        assert_eq!(w[0], 32_749);
        // Strictly decreasing.
        for i in 1..10 {
            assert!(w[i] < w[i - 1], "lag window not strictly decreasing at {i}");
        }
    }

    #[test]
    fn bandwidth_expansion_loads_10_values() {
        let w = bandwidth_expansion();
        assert_eq!(w.len(), 10);
        // First entry close to ~0.994 * 32768 = 32571 (γ ≈ 0.994).
        assert_eq!(w[0], 32_571);
        // Strictly decreasing — γ^i for i = 1..10 with γ ∈ (0, 1).
        for i in 1..10 {
            assert!(w[i] < w[i - 1]);
        }
    }

    #[test]
    fn lsp_cosine_lookup_loads_512_values() {
        let t = lsp_cosine_lookup();
        assert_eq!(t.len(), 512);
        // cos(0) = 1.0 at Q14 = 16384.
        assert_eq!(t[0], 16_384);
        // Quarter-period (k = 128) ⇒ cos(π/2) = 0.
        assert_eq!(t[128], 0);
        // Half-period (k = 256) ⇒ cos(π) = −1.0 at Q14 = −16384.
        assert_eq!(t[256], -16_384);
        // Three-quarter (k = 384) ⇒ cos(3π/2) = 0.
        assert_eq!(t[384], 0);
        // The table contains both positive and negative values
        // (one full period).
        assert!(t.iter().any(|&v| v > 0));
        assert!(t.iter().any(|&v| v < 0));
    }

    #[test]
    fn lsp_dc_predicted_loads_10_values() {
        let t = lsp_dc_predicted_frequencies();
        assert_eq!(t.len(), 10);
        // Frequencies should be strictly increasing (LSP ordering).
        for i in 1..10 {
            assert!(
                t[i] > t[i - 1],
                "DC LSP frequencies not strictly increasing at {i}"
            );
        }
    }

    #[test]
    fn lsp_band_info_partitions_lpc_order_10() {
        let bands = lsp_band_info();
        assert_eq!(bands.len(), 3);
        // (start, length) per band: (0, 3), (3, 3), (6, 4).
        assert_eq!(bands[0], [0, 3]);
        assert_eq!(bands[1], [3, 3]);
        assert_eq!(bands[2], [6, 4]);
        // Bands cover LSP indices 0..10 contiguously.
        let total: i16 = bands.iter().map(|b| b[1]).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn perceptual_weighting_pole_zero_load() {
        assert_eq!(perceptual_weighting_zero().len(), 10);
        assert_eq!(perceptual_weighting_pole().len(), 10);
        // Pole table = γ₂^i with γ₂ = 0.5 (Q15: γ₂^1 = 16 384).
        let pole = perceptual_weighting_pole();
        assert_eq!(pole[0], 16_384);
        assert_eq!(pole[1], 8_192);
        assert_eq!(pole[2], 4_096);
    }

    #[test]
    fn postfilter_pole_zero_load() {
        assert_eq!(postfilter_zero().len(), 10);
        assert_eq!(postfilter_pole().len(), 10);
        // Post-filter pole tap 0 (γ₂ = 0.75) ⇒ 0.75 * 32 768 = 24 576.
        let pole = postfilter_pole();
        assert_eq!(pole[0], 24_576);
    }

    #[test]
    fn highpass_filter_constants_load_2() {
        let t = highpass_filter_constants();
        assert_eq!(t.len(), 2);
        // Hex 0x1800 = 6 144, 0x2000 = 8 192.
        assert_eq!(t[0], 0x1800);
        assert_eq!(t[1], 0x2000);
    }

    #[test]
    fn gain_quantizer_decision_factors_load_4() {
        let t = gain_quantizer_decision_factors();
        assert_eq!(*t, [273, 998, 499, 333]);
    }

    #[test]
    fn bit_allocation_segment_tables_load() {
        let b = bit_allocation_segment_boundaries();
        assert_eq!(*b, [2_048, 18_432, 231_233]);
        let a = bit_allocation_segment_base();
        assert_eq!(*a, [0, 32, 96]);
    }

    #[test]
    fn mp_mlq_pulse_count_load() {
        let t = mp_mlq_pulse_count_per_subframe();
        assert_eq!(*t, [6, 5, 6, 5]);
    }

    #[test]
    fn mp_mlq_max_position_table_load() {
        let t = mp_mlq_max_position_table();
        assert_eq!(t.len(), 4);
        // Hex 0x00090f6f = 593 263. The table has two distinct values
        // alternating across the 4 subframes.
        assert_eq!(t[0], 0x0009_0f6f);
        assert_eq!(t[1], 0x0002_2caa);
        assert_eq!(t[2], 0x0009_0f6f);
        assert_eq!(t[3], 0x0002_2caa);
    }

    #[test]
    fn accessor_caching_returns_same_pointer() {
        // OnceLock should hand back the same `'static` reference on
        // subsequent calls — confirms the parse runs exactly once.
        let a = hamming_window().as_ptr();
        let b = hamming_window().as_ptr();
        assert_eq!(a, b);
    }
}
