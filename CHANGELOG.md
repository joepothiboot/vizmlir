# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-03

### Added

- Live baseline/current MLIR comparison with SSA-renumbering-aware structural diffs.
- Side-by-side MLIR editors with added, removed, and changed operation reporting.

### Fixed

- Restored the Rust lexer and WebAssembly parser build used by the visualizer.
- Fixed canvas and editor selectors so parsing and graph rendering initialize correctly.

## [0.1.0] - 2026-09-01

### Added

- Initial project structure and setup.
- Basic integration for MLIR code rendering.
- Interactive graph visualization interface.
- Core project documentation (`README.md`).
