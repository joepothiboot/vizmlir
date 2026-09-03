# VizMLIR

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight, browser-based visualizer for MLIR. Designed primarily for compiler engineers who need to inspect lowering passes and understand IR flow without the noise of raw text logs.

**[Try VizMLIR live](https://joepotibutr.github.io/vizmlir/)**

## Why VizMLIR?

Parsing complex MLIR by eye is a bottleneck. VizMLIR provides an interactive graph interface to render your IR instantly. It’s built to be a simple, "no-friction" tool that you can keep open in a side tab while you iterate on your compiler passes.

## Who is it for?

- **Compiler engineers:** Inspect MLIR graphs and compare the effect of lowering passes.
- **Frontend engineers:** Explore the React and WebAssembly implementation, or contribute improvements to the UI.
- **Technical readers:** Build an intuition for how MLIR operations connect without needing to read raw dumps alone.

## Features

- **Interactive Graphs:** Visualize nodes and operations as they connect through your IR.
- **Low Latency:** Renders changes in real-time as you modify your MLIR.
- **Browser-Native:** Runs entirely on the client side—no backend infrastructure required.
- **Streamlined UI:** Minimalist interface that stays out of your way.

## Quick Start

To run VizMLIR locally:

```bash
git clone https://github.com/joepotibutr/vizmlir
cd vizmlir
npm install
npm run dev
```

1. Open your browser to the local dev address.
2. Paste your MLIR into the input editor.
3. The graph view will generate automatically. Use the navigation controls to zoom or pan.

## Contributing

Feedback and contributions are highly appreciated. If you encounter a specific MLIR construct that doesn't render as expected, please open an issue with the snippet attached so we can improve the parser.

## License

Distributed under the MIT License. See `LICENSE` for more information.
