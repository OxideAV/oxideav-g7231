//! Saturating 16/32-bit fixed-point basic operators for the G.723.1
//! decode chain.
//!
//! ITU-T G.723.1 §1.5 describes the codec "in terms of bit-exact,
//! fixed-point mathematical operations" over 16-bit words; the OVER /
//! TAME conformance classes deliberately drive sustained Word16
//! saturation chains that a floating-point model cannot track
//! long-range. This module supplies the standard saturating DSP
//! arithmetic those chains need:
//!
//! - `Word16` values are `i16` (Q15 fractions or raw sample units).
//! - `Word32` values are `i32` (Q31 fractions / double-precision
//!   accumulators).
//! - Every operation **saturates** to `[-32768, 32767]` /
//!   `[-2^31, 2^31 − 1]` instead of wrapping.
//! - `l_mult(a, b) = 2·a·b`: the fractional multiply doubles the raw
//!   product so Q15 × Q15 lands in Q31 (the lone overflow case is
//!   `l_mult(-32768, -32768)`, which saturates to `i32::MAX`).
//! - `mac`/`msu` accumulate doubled products with saturation at every
//!   step, so a long convolution that overflows sticks at the rail the
//!   same way a 16-bit DSP accumulator chain does.
//!
//! These semantics are the industry-standard saturating basic-op set
//! for ITU-T fixed-point speech codecs; they are defined here from the
//! arithmetic statements of the Recommendation's clause 1–4 prose (Q15
//! table formats, "16-bit fixed point arithmetic"), not from any
//! reference listing.

/// Saturate a 32-bit value to the 16-bit range.
#[inline]
pub const fn saturate(x: i32) -> i16 {
    if x > i16::MAX as i32 {
        i16::MAX
    } else if x < i16::MIN as i32 {
        i16::MIN
    } else {
        x as i16
    }
}

/// Saturating 16-bit addition.
#[inline]
pub const fn add(a: i16, b: i16) -> i16 {
    saturate(a as i32 + b as i32)
}

/// Saturating 16-bit subtraction.
#[inline]
pub const fn sub(a: i16, b: i16) -> i16 {
    saturate(a as i32 - b as i32)
}

/// Saturating absolute value (`abs_s(-32768) = 32767`).
#[inline]
pub const fn abs_s(a: i16) -> i16 {
    if a == i16::MIN {
        i16::MAX
    } else if a < 0 {
        -a
    } else {
        a
    }
}

/// Saturating negation (`negate(-32768) = 32767`).
#[inline]
pub const fn negate(a: i16) -> i16 {
    if a == i16::MIN {
        i16::MAX
    } else {
        -a
    }
}

/// Fractional Q15 multiply: `(a · b) >> 15`, saturated. The only
/// saturating input pair is `(-32768, -32768)`.
#[inline]
pub const fn mult(a: i16, b: i16) -> i16 {
    saturate((a as i32 * b as i32) >> 15)
}

/// Fractional Q15 multiply with rounding: `(a · b + 2^14) >> 15`,
/// saturated.
#[inline]
pub const fn mult_r(a: i16, b: i16) -> i16 {
    saturate((a as i32 * b as i32 + (1 << 14)) >> 15)
}

/// Saturating left shift (negative `n` shifts right with sign
/// extension).
#[inline]
pub const fn shl(a: i16, n: i32) -> i16 {
    if n <= 0 {
        return shr(a, -n);
    }
    if n >= 15 {
        return if a > 0 {
            i16::MAX
        } else if a < 0 {
            i16::MIN
        } else {
            0
        };
    }
    saturate((a as i32) << n)
}

/// Arithmetic right shift (negative `n` shifts left with saturation).
#[inline]
pub const fn shr(a: i16, n: i32) -> i16 {
    if n < 0 {
        return shl(a, -n);
    }
    if n >= 15 {
        return if a < 0 { -1 } else { 0 };
    }
    a >> n
}

/// Saturating 32-bit addition.
#[inline]
pub const fn l_add(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

/// Saturating 32-bit subtraction.
#[inline]
pub const fn l_sub(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

/// Saturating 32-bit absolute value.
#[inline]
pub const fn l_abs(a: i32) -> i32 {
    if a == i32::MIN {
        i32::MAX
    } else if a < 0 {
        -a
    } else {
        a
    }
}

/// Doubled fractional multiply: `2·a·b` so Q15 × Q15 → Q31. Saturates
/// only for `l_mult(-32768, -32768)`.
#[inline]
pub const fn l_mult(a: i16, b: i16) -> i32 {
    let p = a as i32 * b as i32;
    if p == 0x4000_0000 {
        i32::MAX
    } else {
        p << 1
    }
}

/// Multiply–accumulate: `acc + 2·a·b`, saturated at each step.
#[inline]
pub const fn l_mac(acc: i32, a: i16, b: i16) -> i32 {
    l_add(acc, l_mult(a, b))
}

/// Multiply–subtract: `acc − 2·a·b`, saturated at each step.
#[inline]
pub const fn l_msu(acc: i32, a: i16, b: i16) -> i32 {
    l_sub(acc, l_mult(a, b))
}

/// Deposit a 16-bit word in the high half of a 32-bit word.
#[inline]
pub const fn l_deposit_h(a: i16) -> i32 {
    (a as i32) << 16
}

/// Deposit a 16-bit word in the low half of a 32-bit word (sign
/// extended).
#[inline]
pub const fn l_deposit_l(a: i16) -> i32 {
    a as i32
}

/// High 16 bits of a 32-bit word (truncation toward −∞).
#[inline]
pub const fn extract_h(a: i32) -> i16 {
    (a >> 16) as i16
}

/// Low 16 bits of a 32-bit word (modular).
#[inline]
pub const fn extract_l(a: i32) -> i16 {
    a as i16
}

/// Round the high 16 bits of a 32-bit word: `extract_h(l_add(a,
/// 0x8000))`.
#[inline]
pub const fn round16(a: i32) -> i16 {
    extract_h(l_add(a, 0x8000))
}

/// Saturating 32-bit left shift (negative `n` shifts right).
#[inline]
pub const fn l_shl(a: i32, n: i32) -> i32 {
    if n <= 0 {
        return l_shr(a, -n);
    }
    if n >= 31 {
        return if a > 0 {
            i32::MAX
        } else if a < 0 {
            i32::MIN
        } else {
            0
        };
    }
    match a.checked_shl(n as u32) {
        Some(v) if (v >> n) == a => v,
        _ => {
            if a > 0 {
                i32::MAX
            } else {
                i32::MIN
            }
        }
    }
}

/// Arithmetic 32-bit right shift (negative `n` shifts left with
/// saturation).
#[inline]
pub const fn l_shr(a: i32, n: i32) -> i32 {
    if n < 0 {
        return l_shl(a, -n);
    }
    if n >= 31 {
        return if a < 0 { -1 } else { 0 };
    }
    a >> n
}

/// Number of left shifts needed to normalise a 16-bit word into
/// `[16384, 32767]` (or `[-32768, -16385]`). `norm_s(0) = 0`.
#[inline]
pub const fn norm_s(a: i16) -> i32 {
    if a == 0 {
        return 0;
    }
    let v = if a < 0 { !(a as i32) } else { a as i32 };
    // Leading redundant sign bits below bit 14.
    let mut n = 0;
    let mut x = v;
    while x < 0x4000 && n < 15 {
        x <<= 1;
        n += 1;
    }
    n
}

/// Number of left shifts needed to normalise a 32-bit word into
/// `[2^30, 2^31 − 1]` (or the negative mirror). `norm_l(0) = 0`.
#[inline]
pub const fn norm_l(a: i32) -> i32 {
    if a == 0 {
        return 0;
    }
    let v = if a < 0 { !a } else { a };
    let mut n = 0;
    let mut x = v;
    while x < 0x4000_0000 && n < 31 {
        x <<= 1;
        n += 1;
    }
    n
}

/// Fractional division `a / b` in Q15 for `0 ≤ a ≤ b`, `b > 0`:
/// returns `32767` when `a == b`. The classic 15-step conditional-
/// subtract long division of 16-bit DSPs.
#[inline]
pub const fn div_s(a: i16, b: i16) -> i16 {
    debug_assert!(a >= 0 && b > 0 && a <= b);
    if a == b {
        return i16::MAX;
    }
    let mut num = a as i32;
    let den = b as i32;
    let mut q: i32 = 0;
    let mut i = 0;
    while i < 15 {
        q <<= 1;
        num <<= 1;
        if num >= den {
            num -= den;
            q += 1;
        }
        i += 1;
    }
    q as i16
}

/// Integer square root of a non-negative 64-bit value (floor).
///
/// Used by the fixed-point gain-scaling stages (eq. 47 / eq. 50 square
/// roots of energy ratios): the ratio is pre-scaled into a wide
/// integer so `isqrt(r · 2^(2k))` yields the root in Qk.
#[inline]
pub const fn isqrt64(v: u64) -> u32 {
    if v == 0 {
        return 0;
    }
    let mut x = v;
    let mut r: u64 = 0;
    // Highest power of four ≤ v.
    let mut bit: u64 = 1 << (62 - (v.leading_zeros() as u64 & !1));
    while bit != 0 {
        if x >= r + bit {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    r as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_saturate_at_the_rails() {
        assert_eq!(add(30000, 10000), i16::MAX);
        assert_eq!(add(-30000, -10000), i16::MIN);
        assert_eq!(add(-1, 1), 0);
        assert_eq!(sub(-30000, 10000), i16::MIN);
        assert_eq!(sub(30000, -10000), i16::MAX);
        assert_eq!(sub(100, 42), 58);
    }

    #[test]
    fn abs_and_negate_handle_int_min() {
        assert_eq!(abs_s(i16::MIN), i16::MAX);
        assert_eq!(abs_s(-5), 5);
        assert_eq!(negate(i16::MIN), i16::MAX);
        assert_eq!(negate(7), -7);
    }

    #[test]
    fn mult_is_q15_with_single_saturating_pair() {
        assert_eq!(mult(i16::MIN, i16::MIN), i16::MAX);
        assert_eq!(mult(16384, 16384), 8192); // 0.5 · 0.5 = 0.25
        assert_eq!(mult(-16384, 16384), -8192);
        assert_eq!(mult(32767, 32767), 32766); // truncation
        assert_eq!(mult_r(32767, 32767), 32766);
        assert_eq!(mult_r(3, 16384), 2); // rounds up where mult truncates
        assert_eq!(mult(3, 16384), 1);
    }

    #[test]
    fn l_mult_doubles_and_saturates() {
        assert_eq!(l_mult(i16::MIN, i16::MIN), i32::MAX);
        assert_eq!(l_mult(16384, 16384), 0x2000_0000);
        assert_eq!(l_mult(1, 1), 2);
        assert_eq!(l_mac(5, 1, 1), 7);
        assert_eq!(l_msu(5, 1, 1), 3);
        assert_eq!(l_mac(i32::MAX, 100, 100), i32::MAX);
        assert_eq!(l_msu(i32::MIN, 100, 100), i32::MIN);
    }

    #[test]
    fn shifts_saturate_and_sign_extend() {
        assert_eq!(shl(0x2000, 2), i16::MAX);
        assert_eq!(shl(-0x2000, 2), i16::MIN);
        assert_eq!(shl(3, 2), 12);
        assert_eq!(shl(3, -1), 1);
        assert_eq!(shr(-1, 4), -1);
        assert_eq!(shr(16, 2), 4);
        assert_eq!(shr(5, -2), 20);
        assert_eq!(shr(i16::MAX, 20), 0);
        assert_eq!(shr(i16::MIN, 20), -1);
        assert_eq!(shl(1, 20), i16::MAX);

        assert_eq!(l_shl(0x2000_0000, 2), i32::MAX);
        assert_eq!(l_shl(-0x2000_0000, 2), i32::MIN);
        assert_eq!(l_shl(12, 2), 48);
        assert_eq!(l_shr(-4, 1), -2);
        assert_eq!(l_shr(4, -1), 8);
        assert_eq!(l_shl(1, 40), i32::MAX);
        assert_eq!(l_shr(-1, 40), -1);
    }

    #[test]
    fn extract_round_deposit() {
        assert_eq!(l_deposit_h(0x1234), 0x1234_0000);
        assert_eq!(l_deposit_l(-2), -2);
        assert_eq!(extract_h(0x1234_8000), 0x1234);
        assert_eq!(extract_l(0x1234_8000u32 as i32), -32768);
        assert_eq!(round16(0x1234_8000), 0x1235); // rounds half up
        assert_eq!(round16(0x1234_7FFF), 0x1234);
        assert_eq!(round16(0x7FFF_FFFF), i16::MAX); // saturating round
    }

    #[test]
    fn norms_place_values_in_the_normalised_band() {
        assert_eq!(norm_s(0), 0);
        assert_eq!(norm_s(1), 14);
        assert_eq!(norm_s(0x4000), 0);
        assert_eq!(norm_s(0x3FFF), 1);
        assert_eq!(norm_s(-1), 15);
        assert_eq!(norm_s(i16::MIN), 0);
        for a in [1i16, 2, 5, 100, 3000, 16383, 16384, 32767, -7, -32768] {
            let n = norm_s(a);
            let v = shl(a, n);
            assert!(
                (16384..=32767).contains(&(v as i32)) || (-32768..=-16385).contains(&(v as i32)),
                "norm_s({a}) = {n} → {v}"
            );
        }

        assert_eq!(norm_l(0), 0);
        assert_eq!(norm_l(1), 30);
        assert_eq!(norm_l(i32::MIN), 0);
        for a in [1i32, 77, 0x3FFF_FFFF, 0x4000_0000, i32::MAX, -1, -12345] {
            let n = norm_l(a);
            let v = l_shl(a, n);
            assert!(
                !(-0x4000_0000..0x4000_0000).contains(&v),
                "norm_l({a}) = {n} → {v}"
            );
        }
    }

    #[test]
    fn div_s_is_fractional_long_division() {
        assert_eq!(div_s(1, 1), i16::MAX);
        assert_eq!(div_s(0, 5), 0);
        assert_eq!(div_s(1, 2), 16384);
        assert_eq!(div_s(1, 4), 8192);
        assert_eq!(div_s(3, 4), 24576);
        // Truncating: 1/3 in Q15 is 10922.67 → 10922.
        assert_eq!(div_s(1, 3), 10922);
    }

    #[test]
    fn isqrt64_is_floor_sqrt() {
        assert_eq!(isqrt64(0), 0);
        assert_eq!(isqrt64(1), 1);
        assert_eq!(isqrt64(3), 1);
        assert_eq!(isqrt64(4), 2);
        assert_eq!(isqrt64(1 << 30), 1 << 15);
        assert_eq!(isqrt64(u64::MAX), u32::MAX);
        for v in [2u64, 99, 12345, 1 << 33, (1 << 40) - 1, 987654321987] {
            let r = isqrt64(v) as u64;
            assert!(r * r <= v && (r + 1) * (r + 1) > v, "isqrt64({v}) = {r}");
        }
    }

    #[test]
    fn mac_chain_sticks_at_the_rail_like_a_dsp_accumulator() {
        // A long positive accumulation that overflows must stay pinned
        // at i32::MAX rather than wrapping — the behaviour the OVER
        // conformance class exercises.
        let mut acc = 0i32;
        for _ in 0..4096 {
            acc = l_mac(acc, 32767, 32767);
        }
        assert_eq!(acc, i32::MAX);
        // ... and recover monotonically on the way down.
        acc = l_msu(acc, 32767, 32767);
        assert!(acc < i32::MAX);
    }
}
