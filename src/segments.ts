/**
 * Kind-aware segmentation of tool output: the units the context filter asks about when
 * `context.filter.segments` is on, ported from jev-compaction (jevctx/segments.py, MIT).
 *
 * Splitting on line counts is wrong for most of what an agent reads: a stack trace cut in half is no longer a stack
 * trace, a JSON record cut mid-value is not parseable, a table row without its header is unreadable. So the output is
 * split at boundaries that mean something — whole stack traces (found before anything else, so a trace inside a log
 * survives), diff files and hunks, top-level JSON elements, log stanzas at level changes and blank lines, table row
 * groups (the header travels as metadata, so the scorer sees it without duplicating bytes), code at top-level
 * definitions, prose at paragraph starts.
 *
 * **Losslessness is the invariant**: `segments.map(s => s.text).join("") === text` for every input, by construction —
 * cutters return only segment-start line indices and the tiling is derived from them, so no splitter can lose a byte.
 * A wrong kind is acceptable; a lossy split is not.
 */
import { createHash } from "node:crypto";

export type SegmentKind = "stacktrace" | "json" | "diff" | "table" | "log" | "code" | "text";

export interface Segment {
  kind: SegmentKind;
  /** The segment's exact bytes of the original output. */
  text: string;
  /** 1-based, inclusive line span in the original output. */
  lines: { start: number; end: number };
  /** A table's header rows, when this segment holds data rows that do not repeat them. */
  header?: string | undefined;
  /** Whole but over the token cap (a stack trace): never split, so never asked as one. */
  oversized?: boolean | undefined;
}

/** Splittable spans are cut near this many tokens (characters / 4); a stack trace is never cut. */
export const MAX_SEGMENT_TOKENS = 2000;
/** Table rows and log lines are grouped up to these sizes before the token cap applies. */
const ROWS_PER_SEGMENT = 20;
const LOG_LINES_PER_SEGMENT = 25;

export const segmentTokens = (text: string) => Math.ceil(text.length / 4);

/* ─── Stack traces: found first, never divided ──────────────────────── */

const PY_TRACE_HEADER = /^\s*Traceback \(most recent call last\):\s*$/;
const PY_FRAME = /^\s+File "[^"]*", line \d+/;
const JS_FRAME = /^\s+at\s+\S/;
const JAVA_FRAME = /^\s+at\s+[\w$./]+\(/;
const JAVA_CAUSE = /^\s*(Caused by|\.\.\. \d+ more|Suppressed)/;
const EXC_LINE = /^[\w.]*(Error|Exception|Throwable)\b.*/;
const IS_FRAME = new RegExp([PY_FRAME.source, JS_FRAME.source, JAVA_FRAME.source].map(source => `(?:${source})`).join("|"));

/** Inclusive 0-based line ranges holding a stack trace, so a trace embedded in a log is still kept whole. */
export function findTraceRegions(lines: readonly string[]): Array<{ start: number; end: number }> {
  const regions: Array<{ start: number; end: number }> = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!.replace(/[\r\n]+$/, "");
    let start: number | undefined;
    if (PY_TRACE_HEADER.test(line)) start = i;
    else if (EXC_LINE.test(line) && i + 1 < lines.length && IS_FRAME.test(lines[i + 1]!.replace(/[\r\n]+$/, ""))) start = i;
    else if (IS_FRAME.test(line)) start = i;
    if (start === undefined) { i++; continue; }

    let j = start;
    let lastFrame = start;
    while (j + 1 < lines.length) {
      const next = lines[j + 1]!.replace(/[\r\n]+$/, "");
      if (IS_FRAME.test(next) || JAVA_CAUSE.test(next)) { j++; lastFrame = j; }
      else if (next.trim() === "" && j + 2 < lines.length && IS_FRAME.test(lines[j + 2]!.replace(/[\r\n]+$/, ""))) { j += 2; lastFrame = j; }
      else if (/^[ \t]/.test(next) && next.trim()) j++;
      else if (EXC_LINE.test(next) && lastFrame > start) { j++; lastFrame = j; break; }
      else break;
    }
    if (lastFrame > start || PY_TRACE_HEADER.test(line)) { regions.push({ start, end: lastFrame }); i = lastFrame + 1; }
    else i = start + 1;
  }
  return regions;
}

/* ─── Kind detection ────────────────────────────────────────────────── */

const DIFF_FILE = /^diff --git /;
const DIFF_HUNK = /^@@ -\d+(,\d+)? \+\d+(,\d+)? @@/;
const LOG_PREFIX = /^\s*(?:\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\[?\d{2}:\d{2}:\d{2}|(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b|npm\s|pip\s|yarn\s)/;
const LOG_LEVEL = /\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b/;
const CODE_TOP_LEVEL = /^(?:def |class |async def |function |export |public |private |protected |@[\w.]+\s*$|const \w+ = \(|var \w+ = function)/;
const CODE_EXTENSIONS = new Set([".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".rb", ".php", ".cs", ".swift", ".kt", ".scala"]);

/** Classify a region. `ref` is the file the output shows, when known: its extension says code. */
export function detectKind(text: string, ref?: string): SegmentKind {
  const stripped = text.trim();
  if (!stripped) return "text";
  if ("[{".includes(stripped[0]!) && "]}".includes(stripped.at(-1)!)) {
    try { JSON.parse(stripped); return "json"; } catch { /* not JSON after all */ }
  }
  const lines = text.split("\n").map(line => line.replace(/[\r\n]+$/, ""));
  const nonEmpty = lines.filter(line => line.trim());
  if (!nonEmpty.length) return "text";
  if (lines.some(line => DIFF_FILE.test(line) || DIFF_HUNK.test(line))) return "diff";
  if (findTraceRegions(lines.map(line => `${line}\n`)).length) return "stacktrace";
  if (nonEmpty.length >= 3 && looksTabular(nonEmpty)) return "table";
  if (nonEmpty.filter(line => LOG_PREFIX.test(line)).length / nonEmpty.length >= 0.6) return "log";
  if (typeof ref === "string" && [...CODE_EXTENSIONS].some(extension => ref.split("?")[0]!.endsWith(extension))) return "code";
  if (nonEmpty.filter(line => CODE_TOP_LEVEL.test(line)).length >= 2) return "code";
  return "text";
}

function looksTabular(nonEmpty: readonly string[]): boolean {
  for (const separator of ["|", "\t"]) {
    const counts = nonEmpty.map(line => line.split(separator).length - 1);
    if (Math.min(...counts) >= 1 && new Set(counts).size <= 2) return true;
  }
  return false;
}

/* ─── Per-kind cut points: segment-start line indices only ──────────── */

/** Turn segment-start indices into an exact tiling of `[0, lines)`. */
function rangesFromCuts(cuts: ReadonlySet<number>, lineCount: number): Array<{ start: number; end: number }> {
  const starts = [...new Set([...cuts].filter(cut => cut > 0 && cut < lineCount))].sort((a, b) => a - b);
  if (!starts.length || starts[0] !== 0) starts.unshift(0);
  const bounds = [...starts, lineCount];
  return bounds.slice(0, -1).map((start, index) => ({ start, end: bounds[index + 1]! - 1 }));
}

/** Cut between top-level JSON elements, tracking depth outside string literals. */
function cutsJson(lines: readonly string[]): number[] {
  const cuts: number[] = [];
  let depth = 0;
  let inString = false;
  let escaped = false;
  let entered = false;
  lines.forEach((raw, index) => {
    const startedAtTop = depth <= 1 && entered;
    for (const character of raw) {
      if (inString) {
        if (escaped) escaped = false;
        else if (character === "\\") escaped = true;
        else if (character === '"') inString = false;
        continue;
      }
      if (character === '"') inString = true;
      else if (character === "[" || character === "{") { depth++; entered = true; }
      else if (character === "]" || character === "}") depth--;
    }
    if (startedAtTop && depth >= 1) cuts.push(index);
  });
  return mergeStructural(lines, cuts);
}

/** Drop cuts that would isolate a line of pure JSON punctuation: a lone `[` is not worth a question of its own. */
function mergeStructural(lines: readonly string[], cuts: readonly number[]): number[] {
  const structural = (span: { start: number; end: number }) => !lines.slice(span.start, span.end + 1).join("").replace(/[ \t\r\n\[\]{},:]/g, "");
  const merged: Array<{ start: number; end: number }> = [];
  for (const span of rangesFromCuts(new Set(cuts), lines.length)) {
    if (merged.length && structural(span)) merged[merged.length - 1]!.end = span.end;
    else merged.push({ ...span });
  }
  while (merged.length > 1 && structural(merged[0]!)) { merged[1] = { start: merged[0]!.start, end: merged[1]!.end }; merged.shift(); }
  return merged.slice(1).map(span => span.start);
}

function tableHeaderSize(lines: readonly string[]): number {
  return lines.length > 1 && /^[\s|:+-]+$/.test(lines[1]!.replace(/[\r\n]+$/, "")) ? 2 : 1;
}

function cutsTable(lines: readonly string[], header: number): number[] {
  const cuts: number[] = [];
  for (let index = header; index < lines.length; index += ROWS_PER_SEGMENT) cuts.push(index);
  return cuts;
}

/** Cut at level changes, at blank-line boundaries, and every N lines otherwise: a stanza boundary is where one thing ended. */
function cutsLog(lines: readonly string[]): number[] {
  const cuts: number[] = [];
  let current: string | undefined;
  let sinceCut = 0;
  let previousBlank = false;
  lines.forEach((raw, index) => {
    const level = LOG_LEVEL.exec(raw)?.[1];
    const blank = raw.trim() === "";
    const boundary = (level !== undefined && current !== undefined && level !== current) || (previousBlank && !blank) || sinceCut >= LOG_LINES_PER_SEGMENT;
    if (index && boundary) { cuts.push(index); sinceCut = 0; }
    if (level !== undefined) current = level;
    previousBlank = blank;
    sinceCut++;
  });
  return cuts;
}

function cutsCode(lines: readonly string[]): number[] {
  const cuts = lines.flatMap((line, index) => index && CODE_TOP_LEVEL.test(line.replace(/[\r\n]+$/, "")) ? [index] : []);
  return cuts.length ? cuts : cutsText(lines);
}

/** Paragraph starts: the first non-blank line after a blank run. */
function cutsText(lines: readonly string[]): number[] {
  const cuts: number[] = [];
  let previousBlank = false;
  lines.forEach((raw, index) => {
    const blank = raw.trim() === "";
    if (index && previousBlank && !blank) cuts.push(index);
    previousBlank = blank;
  });
  return cuts;
}

function cutPoints(kind: SegmentKind, lines: readonly string[]): number[] {
  switch (kind) {
    case "stacktrace": return [];
    case "json": return cutsJson(lines);
    case "diff": return lines.flatMap((line, index) => DIFF_FILE.test(line) || DIFF_HUNK.test(line) ? [index] : []);
    case "table": return cutsTable(lines, tableHeaderSize(lines));
    case "log": return cutsLog(lines);
    case "code": return cutsCode(lines);
    case "text": return cutsText(lines);
  }
}

/* ─── Assembly ──────────────────────────────────────────────────────── */

/** A splittable span over the token cap is divided at line boundaries; a stack trace never is. */
function splitOversized(lines: readonly string[], span: { start: number; end: number }, maxTokens: number): Array<{ start: number; end: number }> {
  if (segmentTokens(lines.slice(span.start, span.end + 1).join("")) <= maxTokens) return [span];
  const pieces: Array<{ start: number; end: number }> = [];
  let current = span.start;
  let running = 0;
  for (let index = span.start; index <= span.end; index++) {
    const cost = segmentTokens(lines[index]!);
    if (running && running + cost > maxTokens) { pieces.push({ start: current, end: index - 1 }); current = index; running = 0; }
    running += cost;
  }
  pieces.push({ start: current, end: span.end });
  return pieces;
}

/**
 * Split `text` into scoreable segments without losing a byte. Stack traces are located first and kept whole — an
 * oversized trace is marked `oversized` rather than split, because a partial trace is worse than a large one.
 */
export function segmentOutput(text: string, ref?: string, maxSegmentTokens = MAX_SEGMENT_TOKENS): Segment[] {
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  if (!lines.length) return [];

  // Trace regions first; the gaps between them are segmented by their own detected kind.
  const regions: Array<{ start: number; end: number; trace: boolean }> = [];
  let cursor = 0;
  for (const region of findTraceRegions(lines)) {
    if (region.start > cursor) regions.push({ start: cursor, end: region.start - 1, trace: false });
    regions.push({ ...region, trace: true });
    cursor = region.end + 1;
  }
  if (cursor < lines.length) regions.push({ start: cursor, end: lines.length - 1, trace: false });

  const spans: Array<{ start: number; end: number; kind: SegmentKind; splittable: boolean }> = [];
  for (const region of regions) {
    if (region.trace) { spans.push({ ...region, kind: "stacktrace", splittable: false }); continue; }
    const regionLines = lines.slice(region.start, region.end + 1);
    const kind = detectKind(regionLines.join(""), ref);
    // A stack trace detected inside a region that was not matched whole was already cut by the region boundary.
    const effective = kind === "stacktrace" ? "text" : kind;
    for (const span of rangesFromCuts(new Set(cutPoints(effective, regionLines)), regionLines.length))
      spans.push({ start: region.start + span.start, end: region.start + span.end, kind: effective, splittable: true });
  }

  const first = spans[0];
  const header = first && first.kind === "table" ? lines.slice(first.start, first.start + tableHeaderSize(lines.slice(first.start, first.end + 1))).join("") : undefined;
  const segments: Segment[] = [];
  for (const span of spans) {
    const pieces = span.splittable ? splitOversized(lines, span, maxSegmentTokens) : [span];
    for (const piece of pieces) {
      const body = lines.slice(piece.start, piece.end + 1).join("");
      segments.push({
        kind: span.kind,
        text: body,
        lines: { start: piece.start + 1, end: piece.end + 1 },
        ...(span.kind === "table" && header !== undefined && !body.startsWith(header) ? { header } : {}),
        ...(!span.splittable && segmentTokens(body) > maxSegmentTokens ? { oversized: true } : {}),
      });
    }
  }
  return segments;
}

/** A stable segment id for logs and tests: the text's digest, salted with the ref (the file the output shows). */
export function segmentId(segment: Segment, ref?: string): string {
  return createHash("sha256").update(`${ref ?? ""}\n${segment.text}`).digest("hex").slice(0, 16);
}
