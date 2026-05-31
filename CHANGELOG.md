# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
