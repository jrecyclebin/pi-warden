// Replays the recorded compactions in Pi's session files through the fast-jev compaction strategy, with real Jev
// requests over the configured backend, and compares each result with the summary Pi wrote. Billable: two questions
// per tool call, one request per batch. The sibling of relevance-replay.mjs; the two reports share their shape so the
// strategies compare line for line.
// Build first; run: node scripts/fast-jev-replay.mjs [--dry-run] [--cap 1500] [--out <dir>] [--sessions <dir>]
// The output directory gets counts, sizes, rates, and decisions per compaction; never session text. Keep it private anyway.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { authState, createTypeSafe } from 'pi-typesafe';
import { defaultConfig, fastCompaction, fastCompactionPlan, loadConfig, spanEntries, taskSpine } from '../dist/index.js';
import { judgeOptions } from '../dist/backend.js';

const args = process.argv.slice(2);
const flag = (name, fallback) => { const index = args.indexOf(name); return index >= 0 ? args[index + 1] : fallback; };
const dryRun = args.includes('--dry-run');
// No Pi compaction has ever been recorded on this machine (the big sessions ran on providers that compact on their own),
// so the default mode has nothing to replay. Synthetic mode cuts real sessions where auto-compaction would trigger and
// measures against what the agent actually did next; the re-fetch ground truth is real, only Pi's own summary is absent.
const synthetic = args.includes('--synthetic');
const cap = Number(flag('--cap', synthetic ? '400' : '1500'));
const out = resolve(flag('--out', '.local/fast-jev-replay'));
const sessionsRoot = resolve(flag('--sessions', join(homedir(), '.pi', 'agent', 'sessions')));
const SINGLE_RUNS = 3;
/** A span this big is where auto-compaction plausibly triggers; the cut lands at the first user boundary past it. */
const TARGET_SPAN_TOKENS = Number(flag('--target-tokens', '20000'));
/** Messages that must follow the cut, so the re-fetch window is real follow-up work, not the session's tail end. */
const MIN_AFTER_MESSAGES = 8;
/** The extension's own per-request timeout (config.timeoutMs), so replay latency matches a session. */
const REQUEST_TIMEOUT_MS = defaultConfig().timeoutMs;
const compaction = defaultConfig().compaction;

function walk(dir, files = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path, files);
    else if (entry.name.endsWith('.jsonl')) files.push(path);
  }
  return files;
}

/** The branch Pi resumes: the parent chain from the last entry. */
function mainPath(entries) {
  const byId = new Map(entries.filter(entry => typeof entry?.id === 'string').map(entry => [entry.id, entry]));
  const last = entries.at(-1);
  if (!last || typeof last.id !== 'string' || byId.size === 0) return entries;
  const path = [];
  const seen = new Set();
  for (let current = last; current && !seen.has(current.id); current = current.parentId ? byId.get(current.parentId) : undefined) {
    seen.add(current.id);
    path.push(current);
  }
  return path.reverse();
}

/** The message an entry puts in context, as Pi's compaction sees it. */
function entryMessage(entry) {
  if (entry.type === 'message' && entry.message) return entry.message;
  if (entry.type === 'custom_message') return { role: 'custom', customType: entry.customType, content: entry.content };
  if (entry.type === 'branch_summary') return { role: 'branchSummary', summary: entry.summary };
  return undefined;
}

const toolCalls = messages => messages.filter(message => message.role === 'assistant').flatMap(message => (Array.isArray(message.content) ? message.content : []).filter(block => block?.type === 'toolCall'));
const callKey = block => {
  const input = block.arguments ?? {};
  if (block.name === 'read' && typeof input.path === 'string') return `read:${input.path}`;
  if (block.name === 'bash' && typeof input.command === 'string') return `bash:${input.command.trim()}`;
  return undefined;
};

function fileOps(messages, previous) {
  const ops = { read: new Set(), written: new Set(), edited: new Set() };
  if (previous && !previous.fromHook && previous.details) {
    for (const file of previous.details.readFiles ?? []) ops.read.add(file);
    for (const file of previous.details.modifiedFiles ?? []) ops.edited.add(file);
  }
  for (const block of toolCalls(messages)) {
    const path = block.arguments?.path;
    if (typeof path !== 'string') continue;
    if (block.name === 'read') ops.read.add(path);
    else if (block.name === 'write') ops.written.add(path);
    else if (block.name === 'edit') ops.edited.add(path);
  }
  return ops;
}

/** Every compaction on the resumed branch, with the span it replaced and the first ten tool calls after it. */
function recordedCompactions() {
  const found = [];
  for (const file of walk(sessionsRoot)) {
    const text = readFileSync(file, 'utf8');
    if (!text.includes('"compaction"')) continue;
    const entries = text.split('\n').filter(line => line.trim()).flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } });
    const path = mainPath(entries);
    let previous;
    let previousIndex = -1;
    for (let index = 0; index < path.length; index++) {
      const entry = path[index];
      if (entry.type !== 'compaction') continue;
      const kept = path.findIndex(candidate => candidate.id === entry.firstKeptEntryId);
      let start = 0;
      if (previous) {
        const previousKept = path.findIndex(candidate => candidate.id === previous.firstKeptEntryId);
        start = previousKept >= 0 ? previousKept : previousIndex + 1;
      }
      const spanEntries = path.slice(start, kept >= 0 ? kept : index).filter(candidate => candidate.type !== 'compaction');
      const messages = spanEntries.map(entryMessage).filter(Boolean);
      const after = [];
      for (const later of path.slice(index + 1)) {
        if (later.type !== 'message' || later.message?.role !== 'assistant') continue;
        for (const block of toolCalls([later.message])) if (after.length < 10) after.push(block);
        if (after.length >= 10) break;
      }
      const spanKeys = new Set(toolCalls(messages).map(callKey).filter(Boolean));
      const refetches = after.map(callKey).filter(key => key && spanKeys.has(key));
      const before = path.slice(0, index);
      const turns = before.filter(candidate => candidate.type === 'message' && candidate.message?.role === 'user');
      const lastUser = turns.at(-1)?.message?.content;
      const task = typeof lastUser === 'string' ? lastUser : (Array.isArray(lastUser) ? lastUser.filter(part => part.type === 'text').map(part => part.text).join('\n') : '');
      found.push({
        id: `${createHash('sha256').update(file).digest('hex').slice(0, 8)}-${index}`,
        piSummary: typeof entry.summary === 'string' ? entry.summary : '',
        input: { messages, previousSummary: previous?.summary, fileOps: fileOps(messages, previous), task, spine: taskSpine(before, task) },
        refetches,
      });
      previous = entry;
      previousIndex = index;
    }
  }
  return found;
}

const tokens = chars => Math.ceil(chars / 4);
const quantile = (values, q) => {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  if (q === 0.5) { const mid = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2; }
  return sorted[Math.min(sorted.length - 1, Math.ceil(q * sorted.length) - 1)];
};

/** Whether a re-fetch key's calls survive: whole (kept verbatim), partial (truncated head), or none (dropped). */
function coverage(key, entries, decisions) {
  const [tool, ...rest] = key.split(':');
  const target = rest.join(':');
  const matches = entries.flatMap(entry => entry.calls).filter(call => call.tool === tool
    && (tool === 'read' ? call.input?.path === target : typeof call.input?.command === 'string' && call.input.command.trim() === target));
  if (!matches.length) return 'none';
  // A call with no decision is pinned or unasked: kept, and its result kept with it when the span holds one.
  const actions = matches.map(call => decisions.find(decision => decision.id === call.id)?.action ?? 'keep');
  if (actions.includes('keep')) {
    return matches.some((call, index) => actions[index] === 'keep' && call.result !== undefined) ? 'whole' : 'partial';
  }
  if (actions.includes('drop_result')) return 'partial';
  return 'none';
}

/** One cut per session in synthetic mode: the first user-message boundary whose prefix reaches the target span. */
function syntheticCompactions() {
  const found = [];
  for (const file of walk(sessionsRoot)) {
    const text = readFileSync(file, 'utf8');
    const entries = text.split('\n').filter(line => line.trim()).flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } });
    const path = mainPath(entries);
    const messagesOnPath = path.map((entry, index) => ({ entry, index })).filter(({ entry }) => entry.type === 'message' && entry.message);
    if (messagesOnPath.length < MIN_AFTER_MESSAGES + 4) continue;
    // The span size a message adds, in the same characters-per-four estimate the replay reports tokens with.
    const sizeOf = message => {
      const content = message.content;
      if (typeof content === 'string') return content.length;
      if (!Array.isArray(content)) return 0;
      return content.reduce((sum, block) => sum + (block?.type === 'text' ? String(block.text ?? '').length : block?.type === 'toolCall' ? 80 : 0), 0);
    };
    let chars = 0;
    let boundary;
    let boundarySize;
    for (const { entry, index } of messagesOnPath) {
      if (boundary === undefined && entry.message.role === 'user' && chars / 4 >= TARGET_SPAN_TOKENS) { boundary = index; boundarySize = chars; }
      chars += sizeOf(entry.message);
    }
    if (boundary === undefined) continue; // never reaches the target span: no compaction would have triggered
    const afterCount = messagesOnPath.filter(({ index }) => index > boundary).length;
    if (afterCount < MIN_AFTER_MESSAGES) continue;
    const spanEntries = path.slice(0, boundary).filter(candidate => candidate.type !== 'compaction');
    const messages = spanEntries.map(entryMessage).filter(Boolean);
    if (toolCalls(messages).length < 3) continue; // nothing to score
    const after = [];
    for (const later of path.slice(boundary + 1)) {
      if (later.type !== 'message' || later.message?.role !== 'assistant') continue;
      for (const block of toolCalls([later.message])) if (after.length < 10) after.push(block);
      if (after.length >= 10) break;
    }
    const spanKeys = new Set(toolCalls(messages).map(callKey).filter(Boolean));
    const refetches = after.map(callKey).filter(key => key && spanKeys.has(key));
    const before = path.slice(0, boundary);
    const turns = before.filter(candidate => candidate.type === 'message' && candidate.message?.role === 'user');
    const lastUser = turns.at(-1)?.message?.content;
    const task = typeof lastUser === 'string' ? lastUser : (Array.isArray(lastUser) ? lastUser.filter(part => part.type === 'text').map(part => part.text).join('\n') : '');
    found.push({
      id: `${createHash('sha256').update(file).digest('hex').slice(0, 8)}-s${boundary}`,
      piSummary: '',
      input: { messages, previousSummary: undefined, fileOps: fileOps(messages), task, spine: taskSpine(before, task) },
      refetches,
      spanTokens: Math.ceil(boundarySize / 4),
    });
  }
  return found;
}

const compactions = synthetic ? syntheticCompactions() : recordedCompactions();
const plans = compactions.map(item => {
  // The plan is the same code the run walks, so what it counts is what a session would spend.
  const plan = fastCompactionPlan(item.input, { config: compaction });
  return { item, plan, fits: !plan.tooLarge, candidates: plan.candidates.length, requests: plan.tooLarge ? 0 : plan.batches.length };
});
// The one-call-per-request runs use the spans nearest the median size, so their states look like a typical compaction's.
const medianCandidates = quantile(plans.map(plan => plan.candidates), 0.5);
const singles = plans.filter(plan => plan.fits && plan.candidates > 0).sort((a, b) => Math.abs(a.candidates - medianCandidates) - Math.abs(b.candidates - medianCandidates)).slice(0, SINGLE_RUNS);
const plannedBatched = plans.reduce((sum, plan) => sum + plan.requests, 0);
const plannedSingles = singles.reduce((sum, plan) => sum + plan.candidates, 0);
console.log(`compactions: ${plans.length}; scored calls: ${plans.reduce((sum, plan) => sum + plan.candidates, 0)}; re-fetch calls: ${plans.reduce((sum, plan) => sum + plan.item.refetches.length, 0)}`);
console.log(`planned requests: ${plannedBatched} batched + ${plannedSingles} one-call (${singles.map(plan => plan.candidates).join(', ')} calls) = ${plannedBatched + plannedSingles}; cap ${cap}`);
if (dryRun) process.exit(0);

// The cap is enforced twice: the plan skips what would pass it, and the client refuses request cap + 1.
let budget = cap;
const runBatched = [];
const skipped = [];
for (const plan of plans) {
  if (plan.requests > budget - plannedSingles) { skipped.push({ id: plan.item.id, requests: plan.requests, why: 'cap' }); continue; }
  budget -= plan.requests;
  runBatched.push(plan);
}
const userConfig = loadConfig();
if (userConfig.backendRefusal !== undefined || userConfig.typesafeBackend === undefined) {
  console.error(`judgments are off: the configured backend was refused (${userConfig.backendRefusal ?? 'none configured'}); nothing is sent`);
  process.exit(1);
}
if (!authState({ backend: userConfig.typesafeBackend }).usable) {
  console.error('no usable key for the configured backend; set its key variable or run /typesafe login');
  process.exit(1);
}
const client = createTypeSafe(judgeOptions({ maxRequests: cap, timeoutMs: REQUEST_TIMEOUT_MS, typesafeBackend: userConfig.typesafeBackend }));
let sent = 0;
const judge = { evaluate: (request, options) => { sent++; return client.evaluate(request, options); } };

const rows = [];
const batchedDecisions = new Map();
for (const plan of runBatched) {
  const { item } = plan;
  const result = await fastCompaction(item.input, { judge, config: compaction });
  const decisions = result.ok ? result.decisions : [];
  if (result.ok) batchedDecisions.set(item.id, new Map(decisions.map(decision => [decision.id, decision.action])));
  const refetch = item.refetches.map(key => ({ ours: result.ok ? coverage(key, plan.plan.entries, decisions) : 'fallback', piMentions: item.piSummary.includes(key.slice(key.indexOf(':') + 1)) }));
  rows.push({
    id: item.id, ok: result.ok, reason: result.ok ? undefined : result.reason, detail: result.ok ? undefined : result.detail,
    candidates: plan.candidates, kept: result.stats.kept, resultsDropped: result.stats.resultsDropped, callsDropped: result.stats.callsDropped, pinned: result.stats.pinned,
    requests: result.stats.requests, inputTokens: result.stats.inputTokens, elapsedMs: result.stats.elapsedMs,
    stateTokens: result.stats.stateTokens, stateStage: result.stats.stateStage || null,
    oursTokens: result.ok ? tokens(result.summary.length) : null, piTokens: synthetic ? null : tokens(item.piSummary.length),
    refetch, spanTokens: item.spanTokens ?? null,
  });
  console.log(`${item.id}: ${result.ok ? `kept ${result.stats.kept}/${plan.candidates} (${result.stats.resultsDropped} truncated, ${result.stats.callsDropped} dropped), ~${tokens(result.summary.length)} tokens` : `fallback ${result.reason}`}; ${result.stats.requests} requests, ${result.stats.inputTokens} input tokens, ${result.stats.elapsedMs} ms`);
}

const agreement = [];
for (const plan of singles) {
  const batched = rows.find(row => row.id === plan.item.id);
  if (!batched) { skipped.push({ id: plan.item.id, requests: plan.candidates, why: 'one-call run: batched run skipped' }); continue; }
  // A generous deadline and no per-compaction cap: this run measures decisions, not latency; the client cap still holds.
  // One call per request carries both its questions alone, so batching is the only thing that can differ.
  const single = await fastCompaction(plan.item.input, { judge, config: { ...compaction, timeoutMs: 600_000, maxRequests: Infinity }, questionsPerRequest: 2 });
  const against = batchedDecisions.get(plan.item.id);
  if (!single.ok || !batched.ok || !against) { agreement.push({ id: plan.item.id, ok: false, reason: (single.ok ? undefined : single.reason) ?? 'batched run failed' }); continue; }
  const ids = single.decisions.map(decision => decision.id).filter(id => against.has(id));
  const same = ids.filter(id => single.decisions.find(decision => decision.id === id).action === against.get(id)).length;
  agreement.push({ id: plan.item.id, calls: ids.length, requests: single.stats.requests, ok: true, agree: same });
}

const ran = rows.filter(row => row.requests > 0);
const replaced = rows.filter(row => row.ok);
const refetchAll = rows.flatMap(row => row.refetch);
const count = predicate => refetchAll.filter(predicate).length;
const fallbacks = rows.filter(row => !row.ok).reduce((map, row) => ({ ...map, [row.reason]: (map[row.reason] ?? 0) + 1 }), {});
const report = {
  generatedAt: new Date().toISOString(), mode: synthetic ? 'synthetic' : 'recorded', ...(synthetic ? { targetSpanTokens: TARGET_SPAN_TOKENS } : {}),
  cap, planned: { batched: plannedBatched, singles: plannedSingles }, sent, skipped,
  compactions: plans.length, replayed: rows.length, replaced: replaced.length,
  size: {
    oursMedian: quantile(replaced.map(row => row.oursTokens), 0.5), oursP90: quantile(replaced.map(row => row.oursTokens), 0.9),
    piMedian: synthetic ? null : quantile(replaced.map(row => row.piTokens), 0.5), piP90: synthetic ? null : quantile(replaced.map(row => row.piTokens), 0.9),
    spanMedian: quantile(plans.map(plan => plan.item.spanTokens ?? tokens(JSON.stringify(plan.item.input.messages).length)), 0.5),
    ratioMedian: synthetic ? null : quantile(replaced.map(row => row.oursTokens / Math.max(1, row.piTokens)), 0.5),
  },
  refetch: { total: refetchAll.length, oursWhole: count(r => r.ours === 'whole'), oursPartial: count(r => r.ours === 'partial'), oursNone: count(r => r.ours === 'none'), oursFallback: count(r => r.ours === 'fallback'), piMentions: synthetic ? null : count(r => r.piMentions) },
  decisions: {
    keptMedian: quantile(replaced.map(row => row.kept), 0.5), resultsDroppedMedian: quantile(replaced.map(row => row.resultsDropped), 0.5),
    callsDroppedMedian: quantile(replaced.map(row => row.callsDropped), 0.5), pinnedMedian: quantile(replaced.map(row => row.pinned), 0.5),
  },
  cost: {
    requestsMedian: quantile(ran.map(row => row.requests), 0.5), requestsP90: quantile(ran.map(row => row.requests), 0.9),
    inputTokensMedian: quantile(ran.map(row => row.inputTokens), 0.5), inputTokensP90: quantile(ran.map(row => row.inputTokens), 0.9),
    msMedian: quantile(ran.map(row => row.elapsedMs), 0.5), msP90: quantile(ran.map(row => row.elapsedMs), 0.9),
  },
  fallbacks, agreement, rows,
};
mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'report.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ ...report, rows: undefined }, null, 2));
console.log(`report: ${join(out, 'report.json')}`);
