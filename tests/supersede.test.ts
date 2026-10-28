import assert from "node:assert/strict";
import { test } from "node:test";
import type { Judge } from "pi-typesafe";
import { footprints, relations, ANY_FILE } from "../src/supersede.js";
import type { Footprint, ObservedCall } from "../src/supersede.js";
import { applyUserOverrides, applyProjectOverrides, defaultConfig } from "../src/config.js";
import { relevanceCompaction } from "../src/relevance.js";
import type { RelevanceInput, SpanMessage } from "../src/relevance.js";

const CWD = "/testbed";

const kinds = (tool: string, input: Record<string, unknown>, cwd = CWD) => new Set(footprints(tool, input, cwd).map(print => `${print.kind} ${print.key} ${print.lines ? `${print.lines.start}-${print.lines.end}` : ""}`));
const observe = (id: string, tool: string, input: Record<string, unknown>, cwd = CWD): ObservedCall => ({ id, tool, input });
const made = (calls: readonly ObservedCall[], cwd = CWD) => relations(calls, cwd).map(relation => [relation.older, relation.newer, relation.kind, relation.reason]);

test("footprints: views, writes, runs, and searches", () => {
  assert.deepEqual([...kinds("read", { path: "/testbed/a.py", offset: 10, limit: 20 })], ["view /testbed/a.py 10-29"]);
  assert.ok([...kinds("bash", { command: "sed -n '5,9p' a.py" })].includes("view /testbed/a.py 5-9"), "a sed range is a view");
  assert.ok([...kinds("bash", { command: "cd /testbed/pkg && cat a.py" })].includes("view /testbed/pkg/a.py 1-Infinity"), "cd changes the workdir");
  assert.ok([...kinds("bash", { command: "head -n 40 a.py" })].includes("view /testbed/a.py 1-40"));
  assert.ok([...kinds("read", { path: "/testbed/a.py", offset: 100 })].includes("view /testbed/a.py 100-Infinity"), "a read without a limit covers from its offset on");
  assert.deepEqual([...kinds("edit", { path: "/testbed/a.py", oldString: "x", newString: "y" })], ["write /testbed/a.py "]);
  const writes = new Set(footprints("bash", { command: "sed -i 's/a/b/' a.py b.py && echo hi > out.txt 2>&1" }, CWD).filter(print => print.kind === "write").map(print => print.key));
  assert.deepEqual([...writes], ["/testbed/a.py", "/testbed/b.py", "/testbed/out.txt"]);
  assert.ok(kinds("bash", { command: "git stash" }).has(`write ${ANY_FILE} `));
  assert.ok(!kinds("bash", { command: "git show X | git apply --check" }).has(`write ${ANY_FILE} `), "a dry run applies nothing");
  // A pipeline is not a plain view, and a heredoc's body is not parsed as shell.
  assert.ok(![...kinds("bash", { command: "cat a.py | grep def" })].some(kind => kind.startsWith("view ")));
  assert.deepEqual([...kinds("bash", { command: "python - <<'EOF'\nprint('>' * 3) > 'x'\nEOF" })].map(kind => kind.split(" ")[0]), ["run"]);
  assert.deepEqual(kinds("grep", { pattern: "def f" }), kinds("grep", { pattern: "def f" }), "the same search has the same key");
});

test("relations: a re-run or a covering view supersedes, a narrower view does not", () => {
  const calls = [
    observe("t1", "bash", { command: "python -m pytest tests/test_a.py -q" }),
    observe("r1", "read", { path: "/testbed/a.py", offset: 100, limit: 50 }),
    observe("r2", "read", { path: "/testbed/a.py", offset: 120, limit: 10 }),
    observe("r3", "bash", { command: "cat a.py" }),
    observe("t2", "bash", { command: "cd /testbed &&  python -m pytest tests/test_a.py -q" }),
  ];
  assert.deepEqual(made(calls.slice(0, 3)), [], "a narrower view does not replace a wider one");
  assert.deepEqual(made(calls.slice(0, 4)).map(relation => [relation[0], relation[2]]), [["r1", "superseded"], ["r2", "superseded"]]);
  assert.deepEqual(made(calls.slice(0, 5)).filter(relation => relation[0] === "t1"), [["t1", "t2", "superseded", "the same command was run again"]], "whitespace and a cd prefix do not change the command");
});

test("relations: a write makes views of that file stale, a re-read supersedes the stale copy", () => {
  const calls = [
    observe("r1", "read", { path: "/testbed/a.py" }),
    observe("r2", "read", { path: "/testbed/b.py" }),
    observe("g1", "grep", { pattern: "parse" }),
    observe("e1", "edit", { path: "/testbed/a.py", oldString: "a", newString: "b" }),
    observe("r3", "read", { path: "/testbed/a.py" }),
    observe("s", "bash", { command: "git stash" }),
  ];
  assert.deepEqual(made(calls.slice(0, 4)), [["r1", "e1", "stale", "/testbed/a.py was changed after this output"]], "a grep has no file, a write of one file stales only that file's views");
  assert.deepEqual(made(calls.slice(0, 5)).filter(relation => relation[0] === "r1").map(relation => [relation[1], relation[2]]), [["e1", "stale"], ["r3", "superseded"]], "reading it again after the edit supersedes the stale copy, which stays recorded");
  const stash = made(calls.slice(0, 6)).filter(relation => relation[0] === "r2" || relation[0] === "r3");
  assert.deepEqual(stash.map(relation => [relation[0], relation[2]]), [["r2", "stale"], ["r3", "stale"]], "git stash may have changed every file");
});

test("relations: narrower views together supersede a wider one; trailing filters change nothing", () => {
  const together = [
    observe("r1", "read", { path: "/testbed/a.py", offset: 100, limit: 50 }),
    observe("s1", "bash", { command: "sed -n '90,130p' a.py" }),
    observe("s2", "bash", { command: "sed -n '131,160p' a.py" }),
  ];
  assert.deepEqual(made(together.slice(0, 2)), []);
  assert.deepEqual(made(together).map(relation => [relation[0], relation[2]]), [["r1", "superseded"]]);

  const checks = [
    observe("r4", "read", { path: "/testbed/a.py" }),
    observe("t1", "bash", { command: "python -m pytest tests/t.py 2>&1 | tail -20" }),
    observe("t2", "bash", { command: "git stash && python -m pytest tests/t.py; git stash pop" }),
    observe("l", "bash", { command: "git stash list" }),
    observe("t3", "bash", { command: "python -m pytest tests/t.py | grep FAIL" }),
  ];
  assert.deepEqual(made(checks.slice(0, 3)), [], "checking the old code and restoring the change writes nothing");
  assert.deepEqual(made(checks.slice(0, 4)), []);
  assert.deepEqual(made(checks).filter(relation => relation[0] === "t1").map(relation => [relation[0], relation[2]]), [["t1", "superseded"]], "a trailing filter changes how much of a check is shown, not which check it is");
});

test("relations: calls without input take no part", () => {
  const calls = [
    observe("r1", "read", { path: "/testbed/a.py" }),
    { id: "x1", tool: "read", input: undefined },
    observe("r2", "read", { path: "/testbed/a.py" }),
  ];
  assert.deepEqual(made(calls), [["r1", "r2", "superseded", "/testbed/a.py was viewed again over the same lines"]]);
});

/* ─── Compaction wiring ─────────────────────────────────────────────── */

const call = (id: string, name: string, args: Record<string, unknown>) => ({ type: "toolCall", id, name, arguments: args });
const result = (toolCallId: string, toolName: string, text: string, isError = false): SpanMessage => ({ role: "toolResult", toolCallId, toolName, content: [{ type: "text", text }], isError });
const config = { keepThreshold: 0.5, maxSummaryTokens: 20000, timeoutMs: 2000, maxRequests: 12, supersession: true };

function fakeJudge(scores: Record<string, number> = {}, sent: Array<{ state: unknown; questions: Record<string, unknown> }> = []): Judge {
  return {
    evaluate: (async (request: { state: unknown; questions: Record<string, unknown> }) => {
      sent.push(request);
      return { answers: Object.fromEntries(Object.keys(request.questions).map(id => [id, { type: "noul", noul: scores[id] ?? 0.1 }])), model: "jev-test", usage: { input_tokens: 100, output_tokens: 0 }, elapsedMs: 1 };
    }),
  } as unknown as Judge;
}

test("compaction: a superseded call costs no request and its left-out line says why", async () => {
  const messages: SpanMessage[] = [
    { role: "user", content: "Fix the date parser so the tests pass." },
    { role: "assistant", content: [call("c1", "bash", { command: "npm test" })] },
    result("c1", "bash", "FAIL tests/parse.test.ts\n  expected 2024-01-02", true),
    { role: "assistant", content: [call("c2", "edit", { path: "src/parse.ts", oldString: "split('-')", newString: "split('-').map(Number)" })] },
    result("c2", "edit", "ok"),
    { role: "assistant", content: [call("c3", "bash", { command: "npm test" })] },
    result("c3", "bash", "ok\n3 passing"),
  ];
  const input: RelevanceInput = { messages, task: "Fix the date parser so the tests pass." };
  const sent: Array<{ state: unknown; questions: Record<string, unknown> }> = [];
  const outcome = await relevanceCompaction(input, { judge: fakeJudge({ u2: 0.9, u3: 0.9 }, sent), config });
  assert.ok(outcome.ok);
  assert.deepEqual(Object.keys(sent[0]!.questions), ["u2", "u3"], "the first npm test is not asked about");
  assert.equal(outcome.stats.superseded, 1);
  assert.equal(outcome.stats.candidates, 2);
  assert.match(outcome.summary, /- bash npm test · \d+ chars, failed · the same command was run again/);
  assert.ok(!outcome.summary.includes("expected 2024-01-02"), "the superseded result is not in the summary");
  assert.ok(outcome.summary.includes("3 passing"), "the later result is kept word for word");
});

test("compaction: supersession off by default and in the config parse", async () => {
  assert.equal(defaultConfig().compaction.supersession, false);
  const user = applyUserOverrides(defaultConfig(), JSON.parse('{"compaction": {"supersession": true}}'));
  assert.equal(user.compaction.supersession, true);
  // It sends nothing, so a project file may turn it on too.
  const project = applyProjectOverrides(user, { compaction: { supersession: false } });
  assert.equal(project.compaction.supersession, false);
  const messages: SpanMessage[] = [
    { role: "assistant", content: [call("c1", "bash", { command: "npm test" })] },
    result("c1", "bash", "fail"),
    { role: "assistant", content: [call("c2", "bash", { command: "npm test" })] },
    result("c2", "bash", "ok"),
  ];
  const sent: Array<{ state: unknown; questions: Record<string, unknown> }> = [];
  const outcome = await relevanceCompaction({ messages, task: "t" }, { judge: fakeJudge({}, sent), config: { ...config, supersession: false } });
  assert.ok(outcome.ok);
  assert.deepEqual(Object.keys(sent[0]!.questions), ["u1", "u2"], "with it off every candidate is asked about");
  assert.equal(outcome.stats.superseded, 0);
});

test("compaction: a read made stale by a later edit is pre-pruned, a later re-read of it supersedes the stale copy", async () => {
  const messages: SpanMessage[] = [
    { role: "assistant", content: [call("r1", "read", { path: "src/parse.ts" })] },
    result("r1", "read", "export function parse(s: string) { return s.split('-'); }"),
    { role: "assistant", content: [call("e1", "edit", { path: "src/parse.ts", oldString: "split('-')", newString: "split('-').map(Number)" })] },
    result("e1", "edit", "ok"),
    { role: "assistant", content: [call("r2", "read", { path: "src/parse.ts" })] },
    result("r2", "read", "export function parse(s: string) { return s.split('-').map(Number); }"),
  ];
  const sent: Array<{ state: unknown; questions: Record<string, unknown> }> = [];
  const outcome = await relevanceCompaction({ messages, task: "t" }, { judge: fakeJudge({ u3: 0.9 }, sent), config });
  assert.ok(outcome.ok);
  assert.deepEqual(Object.keys(sent[0]!.questions), ["u2", "u3"], "the stale first read is not asked about");
  assert.match(outcome.summary, /- read src\/parse\.ts · \d+ chars · src\/parse\.ts was viewed again over the same lines/);
});
