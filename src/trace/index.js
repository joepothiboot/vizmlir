// Public surface of src/trace/. Cross-folder imports go through here.
export { analyzeBuffers, bufferTotals, buffersToJSON, compareBuffers, elementBytes, parseMemref } from "./buffers.js";
export { diffStats, lineDiff } from "./linediff.js";
export { countOps, opCountTable, opCountsToCSV, totalOps } from "./opcount.js";
export { embeddedAssembly, findSymbolNode, historyToJSON, scanSymbols, symbolHistory, symbolTimeline } from "./provenance.js";
export { buildLocIndex, findSource, linesOf, nodesAt, opCounts, sameFile, sourcePositions } from "./sources.js";
export { byteLength, extractReports, formatBytes, formatSeconds, matchTiming, timingToJSON } from "./timing.js";
export { baselineFor, describeEvent, isPassTrace, moduleStateAt, parsePassTrace } from "./trace.js";
