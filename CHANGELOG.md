# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Formant-postfilter tilt + adaptive gain scaling reshaped to match
  G.723.1 §3.8 / 3.9** (round 229).
  - The §3.8 tilt-compensation stage `1 − μ · z⁻¹` no longer uses a
    constant `μ = 0.25`. Each subframe now computes the first-order
    normalised autocorrelation `k = r(1)/r(0)` of the synthesis input
    `sy[n]`, smooths it across subframes via the leaky integrator
    `k1 = (1 − POSTFILTER_TILT_SMOOTH_ALPHA) · k1_prev +
    POSTFILTER_TILT_SMOOTH_ALPHA · k` with `α = 1/4`, and applies
    `μ = POSTFILTER_TILT_BASE · k1` (`POSTFILTER_TILT_BASE = 0.25`).
    Silence leaves `μ = 0`; strong low-frequency content pulls `μ` up
    toward `≈ 0.25`. `k1` is bounded to `[−1, 1]` per Cauchy-Schwarz on
    `r(1)/r(0)`.
  - The §3.9 adaptive gain scaling is no longer a per-sample chase with
    `α = 0.85` toward `sqrt(e_in / e_out)`. The spec form is now in
    place: per subframe `g_s = sqrt(Σ sy²[n] / Σ pf²[n])` (set to `1` if
    the denominator is zero, eq. 50); per sample the smoothed gain runs
    as a leaky integrator `g[n] = (1 − α) · g[n − 1] + α · g_s` with
    `α = POSTFILTER_AGC_ALPHA = 1/16` (eq. 51); the output is
    `q[n] = pf[n] · g[n] · (1 + α)` (eq. 52) so the `(1 + 1/16)` boost
    undoes the average attenuation introduced by the integrator.
    `g[−1]` initialises to `POSTFILTER_AGC_INIT_GAIN = 1` per §3.11.
  - Round-trip PSNR on the integration test improves modestly: ACELP
    goes from ~17.4 dB to ~17.6 dB (+0.2 dB); MP-MLQ stays at ~20.7 dB
    inside its ~0.01 dB measurement-floor band. The shape is the
    headline change — tilt now tracks the per-subframe spectral tilt
    instead of cutting at a fixed factor, and the AGC follows the spec's
    leaky-integrator shape with the same `(1 + α)` compensation factor.
  - Five new unit tests pin the new behaviour:
    `post_filter_tilt_k1_smooths_per_subframe_per_spec` drives a low-pass
    synthesis input and verifies `pf_tilt_k1` moves positive on the
    first subframe and stays non-decreasing over six subsequent
    identical subframes while remaining inside `[−1, 1]`;
    `post_filter_tilt_k1_zero_input_zeroes_k` confirms zero input zeros
    `k` and the integrator decays the saved `k1` by `1 − α`;
    `post_filter_agc_holds_unity_on_silence` confirms silence in →
    silence out with the AGC staying at unity;
    `post_filter_agc_leaky_integrator_matches_closed_form` checks the
    per-sample integrator's `SUBFRAME_SIZE`-sample trajectory matches
    the closed form `g[N − 1] = g₀ + (g_s − g₀) · (1 − (1 − α)^N)`;
    `post_filter_state_starts_at_unity_agc` now also pins
    `pf_tilt_k1 = 0` and `pf_agc_gain = POSTFILTER_AGC_INIT_GAIN`.
  - New `tables` constants: `POSTFILTER_TILT_BASE = 0.25`,
    `POSTFILTER_TILT_SMOOTH_ALPHA = 0.25`,
    `POSTFILTER_AGC_ALPHA = 1/16`, `POSTFILTER_AGC_INIT_GAIN = 1.0`,
    all cited to G.723.1 §3.8 / 3.9 / 3.11. The former
    `POSTFILTER_TILT = 0.25` constant is replaced by the
    smoothed-`k1`-driven `μ` so the tilt coefficient is no longer a
    compile-time constant.

- **Frame-erasure concealment reshaped to match G.723.1 §3.10.2** (round
  222). The previous ad-hoc decay schedule (halving the saved gains and
  driving a pseudo-random innovation through the decoder pipeline at
  every erased frame) is replaced by the spec's voiced/unvoiced
  classifier path:
  - The decoder now keeps a saved trailing 120-sample window of
    post-filtered output (`ERASURE_CLASSIFIER_HISTORY_LEN`), the saved
    `L_2` (third-subframe lag), and the saved average of subframes 2
    and 3 fixed-codebook gains.
  - On erasure, a cross-correlation auto-search over `L_2 ± 3`
    (`ERASURE_CLASSIFIER_LAG_RADIUS`) computes the best-lag prediction
    gain in dB. If it exceeds `ERASURE_VOICED_THRESHOLD_DB = 0.58 dB`,
    the frame is classified voiced and concealment regenerates a
    periodic excitation at the classifier's pitch via the adaptive
    codebook with the fixed innovation suppressed; otherwise the frame
    is classified unvoiced and concealment regenerates a uniform
    pseudo-random excitation scaled by the saved average gain.
  - Attenuation follows the spec: 2.5 dB per consecutive erased frame
    (`ERASURE_ATTENUATION_DB_PER_FRAME`), mute completely after 3
    interpolated frames (`ERASURE_MUTE_AFTER_FRAMES`). Frames past the
    mute threshold emit exact silence.
  - LSP extrapolation continues to apply the wider §3.10.1 stability
    procedure (`Δ_min = 62.5 Hz`) on the saved previous-frame LSP, but
    the LSP itself is no longer perturbed by the gain schedule.
  - Two new tests pin the new behaviour:
    `decode_erased_attenuation_schedule_matches_spec` confirms the
    erased-run counter advances and emits exact silence past the mute
    threshold; `erasure_classifier_distinguishes_voiced_and_unvoiced`
    seeds the trailing window with a pure 100 Hz sinusoid (voiced ⇒
    classifier returns lag ≈ 80) and broadband-LCG noise (unvoiced ⇒
    classifier returns voiced = false) and pins the empty-history
    fallback.
  - Both pre-existing integration tests
    (`erasure_in_middle_of_stream_is_concealed`,
    `sustained_erasure_run_decays_to_silence`) continue to pass without
    modification — the spec attenuation schedule still mutes a long
    erasure run well within their 10-frame envelopes.
- **LSP stability check reshaped to match G.723.1 §3.1 / 2.6** (round 216).
  The decoded-LSP post-processing in `dequantise_lsp` is no longer an
  ad-hoc cosine-domain `gap ≥ 0.01` clamp; it now follows the spec's
  procedure (eq. 6–7.3). New `pub(crate) enforce_lsp_stability` operates
  in angular-frequency space: convert cosines → ω via `acos`, find each
  pair `(ω_j, ω_{j+1})` with `ω_{j+1} − ω_j < Δω_min`, spread it around
  its midpoint by `±Δω_min/2`, iterate up to
  `LSP_STABILITY_MAX_ITERATIONS = 10` passes, then re-convert to cosines.
  `Δω_min` is `2π · Δ_min_hz / SAMPLE_RATE_HZ`. The normal path uses
  `Δ_min = 31.25 Hz` (`LSP_STABILITY_DELTA_MIN_HZ`); the erasure
  concealment path now applies the same procedure with the spec's wider
  `Δ_min = 62.5 Hz` (`LSP_STABILITY_DELTA_MIN_ERASURE_HZ`) per §3.10.1,
  pulling the extrapolated previous-frame LSP back into a stable
  configuration when repeated erasures drift its pairs closer together.
  Five new unit tests pin the procedure: already-stable input is a
  no-op, a single inversion converges in one pass with the spreading
  applied around the midpoint, the erasure variant widens minimum gaps
  beyond the normal variant when the input violates the wider floor,
  every dequantised LSP from a probe set of indices hits the 31.25 Hz
  floor and is strictly monotone-decreasing in cosine domain, and an
  all-equal degenerate input still yields a finite LPC via `lsp_to_lpc`.
  No PSNR regression on the round-trip integration test.
- **Pitch (long-term) post-filter reshaped to match G.723.1 §3.6**
  (round 211). The decoder's pitch post-filter is no longer a fixed
  `β = 0.2` LTP at the decoded lag; it now follows the spec shape:
  forward + backward cross-correlations maximised over the seven-lag
  window `M ∈ [L − 3, L + 3]` around the reference lag `L`
  (`L = L_0` covers subframes 0,1 and `L = L_2` covers subframes 2,3
  per §3.6 prose), one-sided weighting `(w_f, w_b) ∈ {(0,0), (0,1),
  (1,0)}` driven by per-side prediction gain (eq. 45–46), a 1.25 dB
  pitch-prediction-gain gate that bypasses the LTP postfilter on
  subframes where it would harm signal quality, and the spec's
  rate-specific LTP weighting `γ_ltp` (0.1875 for the high rate, 0.25
  for the low rate) threaded through a new `pub(crate) Rate {Low,
  High}` enum from each decode entry point. Output energy
  normalisation `g_p ≤ 1` (eq. 47) means the LTP comb cannot inflate
  the subframe energy past the synthesis input. Six new structural
  unit tests pin the gate behaviour (silence + white-noise bypass,
  periodic-signal engagement, rate-dependent deviation, forward and
  backward search lock onto a sinusoid's period). Headline
  integration-test PSNR is preserved (~17.4 dB ACELP / ~20.7 dB
  MP-MLQ on the synthetic voiced signal); the postfilter is now
  signal-adaptive instead of applying a single fixed β to every
  subframe.
- New `tables` constants: `POSTFILTER_LTP_GAMMA_HIGH = 0.1875`,
  `POSTFILTER_LTP_GAMMA_LOW = 0.25`,
  `POSTFILTER_LTP_PRED_GAIN_DB_MIN = 1.25`,
  `POSTFILTER_LTP_SEARCH_RADIUS = 3`, all cited to G.723.1 §3.6.

### Added

- Three Criterion bench harnesses (`benches/encode.rs`,
  `benches/decode.rs`, `benches/roundtrip.rs`) covering both dual rates.
  Inputs are synthesised in-bench from a deterministic sum-of-sinusoids
  generator so the encoder takes the speech-like pitch path rather than
  a near-silent shortcut; no `docs/` fixtures or external files are
  read. Each harness exposes 3–4 scenarios (per-rate, voiced vs silence,
  1 s / 5 s durations, plus a mixed-rate dispatch scenario in
  `decode.rs`). Headline baseline (macOS aarch64, single-thread,
  release): ~ 22 ms/s encode at either rate, ~ 170 µs/s decode at either
  rate, ~ 20 ms/s round-trip — well above real-time at 8 kHz. Pinned to
  `criterion = "0.5"` to match the rest of the OxideAV bench crates.
  Run with `cargo bench -p oxideav-g7231 --bench {encode,decode,roundtrip}`.
- `spec_tables` module exposing the 27 ITU-T G.723.1 normative numeric
  tables (§2.2 high-pass; §2.4 LPC primitives — 180-pt Hamming, 10-pt
  binomial lag, bandwidth-expansion γ^i, 512-pt LSP cosine lookup;
  §2.6 LSP split-VQ DC predictor + 3-band codebooks Band0/1/2 in Q13;
  §2.9 perceptual-weighting filter; §2.13 MP-MLQ pulse counts /
  max-position / 6×30 combinatorial / FCB gain; §2.14 adaptive-codebook
  gain at both rates + decision factors; §2.16 1-tap LTP selector +
  gain; §2.17 taming gain at both rates; §2.18 postfilter; bit-allocation
  segment base + boundaries). Each `pub const [iN; M]` carries a
  doc-comment naming its source CSV under `docs/audio/g7231/tables/`
  and the SHA-256 of the data. Compile-time `const _` asserts pin every
  table's length; 17 unit tests pin structural invariants (Hamming
  symmetry, LSP cosine antisymmetry, FCB-gain monotonicity, LSP
  3-band partition summing to LpcOrder=10, MP-MLQ 6/5/6/5 pulse pattern,
  bit-allocation {0,32,96} / {2048,18432,231233} constants, paired
  LTP selector + gain dimensions, taming-gain floor of 1024). Data
  lives alongside (does not yet replace) the existing internally-
  consistent `tables` codebooks driving the encoder. Threading this
  spec data through the LPC / LSP / gain quantiser to produce a
  bit-exact spec-compatible bitstream is the next-round task.

## [0.0.7](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.6...v0.0.7) - 2026-05-29

### Other

- drop dead synthesis module scaffold

### Removed

- drop dead `synthesis` module — the standalone `LpcSynthesis` scaffold
  was never wired in (the real LPC synthesis filter lives inline inside
  `encoder::SynthesisState`, used by both the encoder analysis-by-
  synthesis loop and the registered decoder). Removing the unused
  `pub mod synthesis` trims the public API surface and the misleading
  "scaffold" docstring.


## [0.0.6](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.5...v0.0.6) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-g7231/pull/502))
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- drop unused SampleFormat / TimeBase imports (slim-frame leftover)
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

### Changed

- **`register` entry point unified on `RuntimeContext`** (task #502).
  The legacy `pub fn register(reg: &mut CodecRegistry)` is renamed to
  `register_codecs` and a new `pub fn register(ctx: &mut
  oxideav_core::RuntimeContext)` calls it internally. Breaking change
  for direct callers passing a `CodecRegistry`; switch to either the
  new `RuntimeContext` entry or the explicit `register_codecs` name.

## [0.0.5](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- add decoder post-filter + frame-erasure concealment
- update encoder module docstring to match current pipeline
- README reflects full-synthesis decoder + round-trip PSNR
- joint gain-pair refinement + MP-MLQ coord-descent tidy
- promote stateful decoder + coordinate-descent pulse search
- fix lsp_to_lpc p/2 buffer truncation + rework encoder-decoder sync
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"

## [0.0.4](https://github.com/OxideAV/oxideav-g7231/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- claim WAVEFORMATEX tag via oxideav-codec CodecTag registry
- fix inverted doc statement about synthesis path
