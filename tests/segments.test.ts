import assert from "node:assert/strict";
import { test } from "node:test";
import type { Judge } from "pi-typesafe";
import { applyProjectOverrides, applyUserOverrides, defaultConfig } from "../src/config.js";
import { buildFilterBatches, filterOutput, FILTER_REQUEST_BYTES } from "../src/filter.js";
import { detectKind, findTraceRegions, segmentId, segmentOutput, segmentTokens, MAX_SEGMENT_TOKENS } from "../src/segments.js";

const PY_TRACE = `Traceback (most recent call last):
  File "src/parse.py", line 12, in parse
    return s.split("-")
  File "tests/test_parse.py", line 8, in test_parse
    assert parse("2024-01-02") == expected
ValueError: invalid literal for int() with base 10: '2024-01-02'
`;

test("segmentation is lossless, whatever the input and the size cap", () => {
  const inputs = [
    "", "\n", "   ", "\n\n\n", "no trailing newline",
    "crlf line one\r\ncrlf line two\r\n",
    "x".repeat(20_000),
    "a\n".repeat(500),
    "﻿{\"k\": 1}\n",
    "中文输出的一行\n另一行中文\n",
    PY_TRACE,
    `${"INFO start\nok\n".repeat(50)}${PY_TRACE}{"records": [${Array.from({ length: 30 }, (_, i) => `{"id": ${i}, "ok": true}`).join(", ")}]}`,
  ];
  for (const text of inputs) {
    for (const cap of [4, 50, MAX_SEGMENT_TOKENS]) {
      const segments = segmentOutput(text, undefined, cap);
      assert.equal(segments.map(segment => segment.text).join(""), text, `lossless at cap ${cap}`);
    }
  }
});

test("kinds are detected: json, diff, stacktrace, table, log, code, text", () => {
  assert.equal(detectKind(`{"records": [${Array.from({ length: 5 }, (_, i) => `{"id": ${i}}`).join(", ")}]}`), "json");
  assert.equal(detectKind("diff --git a/x b/x\n@@ -1,2 +1,2 @@\n-a\n+b\n"), "diff");
  assert.equal(detectKind(PY_TRACE), "stacktrace");
  assert.equal(detectKind("| package | version |\n| --- | --- |\n| left-pad | 1.3 |\n| express | 5.0 |\n| chalk | 5.4 |\n"), "table");
  assert.equal(detectKind(Array.from({ length: 10 }, (_, i) => `2024-01-02 10:0${i} INFO step ${i} done`).join("\n")), "log");
  assert.equal(detectKind("def one():\n    pass\ndef two():\n    pass\n"), "code");
  assert.equal(detectKind("def one():\n    pass\ndef two():\n    pass\n", "src/parse.py"), "code", "the file's extension says code");
  assert.equal(detectKind("Just a paragraph of prose.\n\nAnother paragraph.\n"), "text");
});

test("a stack trace is exactly one segment, an oversized one is marked rather than split", () => {
  const whole = segmentOutput(PY_TRACE);
  assert.equal(whole.length, 1);
  assert.equal(whole[0]!.kind, "stacktrace");
  const giant = segmentOutput(`Traceback (most recent call last):\n${Array.from({ length: 4000 }, (_, i) => `  File "f.py", line ${i}, in g\n    pass\n`).join("")}LastError: still going\n`);
  const trace = giant.find(segment => segment.kind === "stacktrace");
  assert.ok(trace, "the trace is found");
  assert.equal(trace!.oversized, true, "an oversized trace is marked, not split");
  assert.equal(trace!.text.includes("LastError: still going"), true, "and it is whole");
});

test("a trace embedded in a log survives whole, and the log is stanzaed", () => {
  const text = `${Array.from({ length: 8 }, (_, i) => `INFO step ${i} ok`).join("\n")}\n${PY_TRACE}2024-01-02 10:31 ERROR after the failure\n`;
  const segments = segmentOutput(text);
  const traces = segments.filter(segment => segment.kind === "stacktrace");
  assert.equal(traces.length, 1);
  assert.equal(traces[0]!.text, PY_TRACE);
  assert.ok(segments.some(segment => segment.kind === "log"), "the log around the trace is still segmented as a log");
});

test("a diff keeps its file and hunk headers; a JSON array splits per element, each parseable", () => {
  const diff = segmentOutput("diff --git a/x b/x\n@@ -1,2 +1,3 @@\n-a\n+b\nc\n@@ -9,1 +10,2 @@\n-d\n+e\n+f\n");
  const hunks = diff.filter(segment => segment.kind === "diff");
  assert.deepEqual(hunks.map(segment => segment.text.trimStart().split("\n")[0]), ["diff --git a/x b/x", "@@ -1,2 +1,3 @@", "@@ -9,1 +10,2 @@"], "every segment starts at its own header");

  const array = `[${Array.from({ length: 3 }, (_, i) => `{"id": ${i + 1}, "ok": true}`).join(",\n")}\n]`;
  const elements = segmentOutput(array).filter(segment => segment.kind === "json");
  assert.equal(elements.length, 3);
  for (const element of elements) assert.equal(JSON.parse(element.text.trim().replace(/^[\[\],\s]+|[\[\],\s]+$/g, "")).ok, true);
});

test("a table's header travels as metadata, never duplicated into the rows", () => {
  const rows = Array.from({ length: 30 }, (_, i) => `| pkg-${i} | ${i}.0.0 |`).join("\n");
  const table = `| package | version |\n| --- | --- |\n${rows}\n`;
  const segments = segmentOutput(table);
  const dataRows = segments.filter(segment => segment.kind === "table" && segment.text.includes("pkg-"));
  assert.ok(dataRows.length >= 2, "30 rows group into several segments");
  for (const segment of dataRows) {
    assert.equal(segment.text.includes("| package"), false, "the header is not in the rows");
    assert.ok(segment.header !== undefined && segment.header.startsWith("| package"), "but it travels with them");
  }
});

test("line spans are contiguous and reconstruct the original; ids are stable and vary with the ref", () => {
  const text = `Intro paragraph.\n\n${PY_TRACE}log line one\nlog line two\n\nSecond paragraph.\n`;
  const segments = segmentOutput(text);
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  assert.equal(segments[0]!.lines.start, 1);
  assert.equal(segments.at(-1)!.lines.end, lines.length);
  for (let index = 1; index < segments.length; index++) assert.equal(segments[index]!.lines.start, segments[index - 1]!.lines.end + 1);
  for (const segment of segments) assert.equal(lines.slice(segment.lines.start - 1, segment.lines.end).join(""), segment.text);

  const again = segmentOutput(text);
  assert.deepEqual(again.map(segment => segmentId(segment)), segments.map(segment => segmentId(segment)));
  assert.notDeepEqual(segments.map(segment => segmentId(segment, "src/x.py")), segments.map(segment => segmentId(segment)), "the ref salts the id");
});

test("oversized splittable spans are split at the token cap, each piece within it", () => {
  const text = Array.from({ length: 2000 }, (_, i) => `log line ${i}`).join("\n");
  const segments = segmentOutput(text, undefined, 100);
  assert.ok(segments.length > 1);
  for (const segment of segments) assert.ok(segmentTokens(segment.text) <= 100 + 2, `piece within the cap (got ${segmentTokens(segment.text)})`);
});

/* ─── Filter wiring ─────────────────────────────────────────────────── */

type Request = { state: Record<string, unknown>; questions: Record<string, { type: string; criteria: unknown }> };

/** Rates a passage from its state field: a bare string for a size chunk, an object for a segment. */
function fakeJudge(rate: (view: unknown) => number, sent: Request[] = []): Judge {
  return {
    async evaluate(request: unknown) {
      const body = request as Request;
      sent.push(body);
      const answers = Object.fromEntries(Object.keys(body.questions).map(id => [id, { type: "score", score: rate(body.state[id]), confidence: 0.9, legend: {}, probabilities: {} }]));
      return { model: "jev-test", elapsedMs: 1, answers, usage: { input_tokens: 1, output_tokens: 0 } } as never;
    },
  };
}
const viewText = (view: unknown) => typeof view === "string" ? view : String((view as { text: string }).text);

const markedLog = `${PY_TRACE}${Array.from({ length: 40 }, (_, i) => `INFO step ${i} ok for the parser`).join("\n")}\nINFO done\n`;

test("with segments on, a trace is asked whole in one question labelled with its kind", async () => {
  const segments = { ...defaultConfig().context.filter, enabled: true, segments: true };
  const sent: Request[] = [];
  const outcome = await filterOutput(markedLog, { tool: "bash", command: "pytest", task: "Fix the parse error." }, { config: segments, judge: fakeJudge(view => viewText(view).includes("ValueError") ? 3 : 0, sent) });
  assert.ok(outcome.ok);
  const traceView = sent.flatMap(request => Object.entries(request.state)).find(([, view]) => (view as { kind?: string }).kind === "stacktrace");
  assert.ok(traceView, "the trace is one state field of its own");
  const traceQuestion = sent.flatMap(request => Object.values(request.questions)).find(question => String(question.type) === "score");
  assert.match(String((traceQuestion as unknown as { instructions: string }).instructions), /a stack trace, whole/);
  assert.ok(outcome.text.includes(PY_TRACE), "a kept trace is kept whole");
  assert.ok(outcome.text.includes("lines omitted"), "gaps are marked");
  assert.equal(outcome.kept, 1, "only the scored trace is chosen; the rest rides the tail or the gaps");
});

test("a table group's state carries the header; a segment that cannot fit a request is never asked", () => {
  const rows = Array.from({ length: 40 }, (_, i) => `| pkg-${i} | ${i}.0.0 |`).join("\n");
  const table = `| package | version |\n| --- | --- |\n${rows}\n`;
  const chunks = segmentOutput(table).map(segment => ({ text: segment.text, kind: segment.kind, ...(segment.header !== undefined ? { header: segment.header } : {}) }));
  const batches = buildFilterBatches(chunks, { tool: "bash", command: "list", task: "t" });
  const views = batches.flatMap(batch => Object.values(batch.state)).filter((view): view is { kind?: string; header?: string; text: string } => typeof view === "object");
  const laterRowGroup = views.find(view => view.kind === "table" && !view.text.includes("| package"));
  assert.ok(laterRowGroup, "a row group after the first repeats the header as context");
  assert.equal(laterRowGroup!.header, "| package | version |\n| --- | --- |\n");
  assert.ok(views.some(view => view.kind === "table" && view.text.startsWith("| package")), "the first group carries the header lines themselves");

  // Redaction can grow text, so the fit is checked on the serialized request: a lone 200k-character trace is out.
  const giant = { text: `${PY_TRACE}${"    frame\n".repeat(20_000)}`, kind: "stacktrace" as const };
  const alone = buildFilterBatches([giant, { text: "small\n" }], { tool: "bash", command: "c", task: "t" }, FILTER_REQUEST_BYTES);
  assert.deepEqual(alone.flatMap(batch => batch.ids), [1], "only the small chunk is asked");
});

test("segments config: off by default, project may tune it, user may turn it on", () => {
  assert.equal(defaultConfig().context.filter.segments, false);
  const user = applyUserOverrides(defaultConfig(), { context: { filter: { segments: true } } });
  assert.equal(user.context.filter.segments, true);
  const project = applyProjectOverrides(defaultConfig(), { context: { filter: { segments: true } } });
  assert.equal(project.context.filter.segments, true, "it sends nothing a chunks filter would not, so a project may set it");
});

test("findTraceRegions finds python, js, and java traces", () => {
  const js = `Error: cannot read properties of undefined\n    at parse (/app/src/parse.js:12:19)\n    at run (/app/tests/x.js:8:3)\n`;
  const java = `Exception in thread "main" java.lang.NullPointerException\n\tat app.Parse.parse(Parse.java:12)\n\tat app.Main.main(Main.java:5)\n`;
  for (const text of [PY_TRACE, js, java]) {
    const regions = findTraceRegions(text.match(/[^\n]*\n|[^\n]+$/g) ?? []);
    assert.equal(regions.length, 1);
    assert.ok(regions[0]!.end > regions[0]!.start);
  }
});
