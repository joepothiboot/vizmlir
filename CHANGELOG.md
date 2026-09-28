# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Read `-mlir-timing` reports (tree, list, and JSON output) from pass traces and pasted logs. Each pass in the timeline shows its wall time and IR size, and **p** opens the full report with links back to each pass.
- Show whole-process peak memory when the log includes `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) output. mlir-opt does not measure memory per pass.
- Download the timing report, peak memory, and per-pass IR sizes as JSON.
- Document the accepted trace format and the parsed structure in `docs/trace-format.md`.
- Test traces from out-of-tree `MlirOptMain` drivers with custom dialects (a `schema-opt` fixture), including namespace-qualified pass names.

### Fixed

- Label ops whose operands contain `=` by their op name: `scf.for %i = %c0 to ..` was labeled `%c0`, and `linalg.generic {indexing_maps = ..}` was labeled `[`. Values bound with `%x =` inside an op (induction variables, `iter_args`) are now definitions.
- Stop treating a region op's trailing types (`} -> tensor<..>`, `} : ..`) as a separate `->` op.

## [0.3.0] - 2026-09-26

### Added

- Load `mlir-opt -mlir-print-ir-after-all` / `-mlir-print-ir-before-all` logs via **Open…** or by pasting, and step through each pass dump with its baseline IR.
- Mark failed passes and show compiler errors, warnings, notes, and remarks printed alongside the dumps.
- Rebuild module baselines from nested function dumps when `-mlir-print-ir-module-scope` is not used.

### Fixed

- `#alias = ...` definitions and trailing `} loc(...)` no longer appear as operations in the graph or diff.

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
