/**
 * Supersession: which earlier tool outputs a later call made obsolete, from the calls' arguments alone. Ported from
 * jev-compaction (jevctx/supersede.py, MIT) for the relevance compaction's pre-pruning: a call in the span that re-ran
 * the same command, re-read the same lines, or was followed by an edit of the file it showed is dropped before any Jev
 * request is spent on it. Deterministic, session-local, no requests, no stored text.
 *
 * A *view* is a read of a file or a range of its lines; a *write* touches a file (or, for `git stash` and friends, every
 * file); a *run* is a command by its normalised text; a *search* is a call's arguments. An earlier output is
 * *superseded* when a later one shows the same thing, and *stale* when a later write touched a file it showed. Several
 * narrower views can together cover a wider one; a write between a view and a later covering view does not undo the
 * covering (the newer view is the fresh text, which is what matters).
 */
import { isAbsolute, join, normalize } from "node:path";
import { commandOf } from "./tools.js";

/** A write that could have changed anything: `git stash`, `git reset`, a whole-tree `patch` … */
export const ANY_FILE = "*";
/** Lines 1 to unbounded: a read with no limit. */
const WHOLE = Number.POSITIVE_INFINITY;

const VIEW_TOOLS = new Set(["read", "view", "cat", "read_file", "view_file"]);
const WRITE_TOOLS = new Set(["edit", "write", "multiedit", "patch", "apply_patch", "str_replace", "edit_file", "write_file"]);
const SEARCH_TOOLS = new Set(["grep", "glob", "list", "ls", "find", "search"]);
const RUN_TOOLS = new Set(["bash", "shell", "powershell", "user bash"]);

export type FootprintKind = "view" | "write" | "run" | "search";

export interface Footprint {
  kind: FootprintKind;
  /** The file (view, write), the normalised command with its workdir (run), or the tool with its arguments (search). */
  key: string;
  /** Lines viewed, 1-based and inclusive; undefined for a whole-file view. View footprints only. */
  lines?: { start: number; end: number } | undefined;
}

export interface Relation {
  /** The unit superseded or made stale. */
  older: string;
  /** The unit that superseded or staled it. */
  newer: string;
  kind: "superseded" | "stale";
  reason: string;
}

/* ─── Footprints ────────────────────────────────────────────────────── */

/** A relative path is resolved against `cwd`; the key compares, it never touches the filesystem. */
function pathKey(value: unknown, cwd: string | undefined): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  const path = value.trim();
  return normalize(isAbsolute(path) ? path : join(cwd ?? ".", path));
}

function readLines(args: Record<string, unknown>): { start: number; end: number } {
  // Pi and OpenCode: 1-based `offset`, `limit` lines. A read without a limit stops at the harness's cap, so it covers from its offset on.
  const offset = typeof args.offset === "number" && args.offset > 0 ? Math.floor(args.offset) : 1;
  const limit = typeof args.limit === "number" && args.limit > 0 ? Math.floor(args.limit) : undefined;
  return { start: offset, end: limit ? offset + limit - 1 : WHOLE };
}

const CD_PREFIX = /^\s*cd\s+(\S+)\s*(?:&&|;)\s*/;
const SED_RANGE = /^(\d+)(?:,(\d+))?p$/;
const STDERR = /\s(?:2>&1|2>\s*\/dev\/null)(?=\s|$)/g;
const FILTER = /\s*\|\s*(?:head|tail|grep|sed|sort|uniq|wc|cat|less)\b[^|\n]*$/;

/** Whitespace collapsed, stderr merges and trailing output filters (`| tail -20`, `| grep FAIL`) dropped: they change how much of a check is shown, not which check it is. A heredoc's body is kept whole. */
function normalise(command: string): string {
  const cut = command.indexOf("\n");
  const firstLine = cut >= 0 ? command.slice(0, cut) : command;
  const heredoc = firstLine.includes("<<");
  const head = heredoc ? firstLine : command;
  const body = heredoc ? command.slice(firstLine.length) : "";
  let text = ` ${head}`.replace(STDERR, " ");
  while (FILTER.test(text)) text = text.replace(FILTER, "");
  return `${text}${body}`.split(/\s+/).join(" ").trim();
}

const PUNCTUATION = ";&|<>";

/** One command line tokenised, quotes respected, shell punctuation (`&&`, `|`, `>`, `;;` …) its own token; undefined when it does not tokenise (an unbalanced quote). */
function tokenize(line: string): string[] | undefined {
  const tokens: string[] = [];
  let word = "";
  let punct = "";
  let quote: string | undefined;
  const endWord = () => { if (word) { tokens.push(word); word = ""; } };
  const endPunct = () => { if (punct) { tokens.push(punct); punct = ""; } };
  for (const character of line) {
    if (quote) { if (character !== quote) word += character; else quote = undefined; continue; }
    if (character === "'" || character === '"') { endWord(); endPunct(); quote = character; continue; }
    if (/\s/.test(character)) { endWord(); endPunct(); continue; }
    if (PUNCTUATION.includes(character)) { endWord(); punct += character; continue; }
    endPunct(); word += character;
  }
  if (quote) return undefined;
  endWord(); endPunct();
  return tokens;
}

const SEPARATORS = new Set(["&&", "||", ";", "|", "&", ";;"]);

/** The command split on `&&`, `||`, `;` and `|`, each tokenised; undefined when it does not tokenise. A heredoc's body is data, not shell: only the line that starts it is parsed. */
function simpleCommands(command: string): string[][] | undefined {
  const cut = command.indexOf("\n");
  const head = command.slice(0, cut >= 0 ? cut : undefined);
  const tokens = tokenize(head.includes("<<") ? head : command);
  if (!tokens) return undefined;
  const parts: string[][] = [[]];
  for (const token of tokens) {
    if (SEPARATORS.has(token)) parts.push([]);
    else parts.at(-1)!.push(token);
  }
  const found = parts.filter(part => part.length > 0);
  return found.length ? found : undefined;
}

/** `cat f`, `nl -ba f`, `head -n N f`, `sed -n 'A,Bp' f`: a view of one file. A pipeline is not a plain view. */
function viewOf(words: readonly string[], cwd: string | undefined): Footprint | undefined {
  const [name, ...rest] = words;
  const files = rest.filter(word => !word.startsWith("-"));
  if (name === "cat" && files.length === 1 && rest.every(word => word === files[0] || word === "-n" || word === "-A"))
    return { kind: "view", key: pathKey(files[0], cwd)!, lines: { start: 1, end: WHOLE } };
  if (name === "nl" && files.length === 1)
    return { kind: "view", key: pathKey(files[0], cwd)!, lines: { start: 1, end: WHOLE } };
  if (name === "head") {
    let count = 10;
    const targets: string[] = [];
    for (let i = 0; i < rest.length; i++) {
      const word = rest[i]!;
      if (word === "-n" && i + 1 < rest.length && /^\d+$/.test(rest[i + 1]!)) { count = Number(rest[i + 1]); i++; continue; }
      if (/^-n?\d+$/.test(word)) count = Number(word.replace(/^-n?/, ""));
      else if (!word.startsWith("-")) targets.push(word);
    }
    if (targets.length === 1) return { kind: "view", key: pathKey(targets[0], cwd)!, lines: { start: 1, end: count } };
    return undefined;
  }
  if (name === "sed" && rest[0] === "-n" && rest.length === 3) {
    const match = SED_RANGE.exec(rest[1]!.replace(/^['"]|['"]$/g, ""));
    const key = pathKey(rest[2], cwd);
    if (match && key) return { kind: "view", key, lines: { start: Number(match[1]), end: Number(match[2] ?? match[1]) } };
  }
  return undefined;
}

const GIT_ANY_SUBCOMMANDS = ["reset", "apply", "am", "pull", "merge", "rebase", "cherry-pick"];

/** Files one simple command may have changed; dry runs and reports (`--check`, `--stat` …) apply nothing. */
function writesOf(words: readonly string[], cwd: string | undefined): string[] {
  const [name, ...rest] = words;
  const paths: Array<string | undefined> = [];
  if (name === "sed" && rest.some(word => word.startsWith("-i") || word === "--in-place")) {
    const operands = rest.filter(word => !word.startsWith("-"));
    const scriptGiven = rest.includes("-e") || rest.includes("-f");
    paths.push(...(scriptGiven ? operands : operands.slice(1)).map(word => pathKey(word, cwd)));
  } else if (name === "tee") paths.push(...rest.filter(word => !word.startsWith("-")).map(word => pathKey(word, cwd)));
  else if (name === "cp" || name === "mv" || name === "install") {
    const operands = rest.filter(word => !word.startsWith("-"));
    if (operands.length >= 2) {
      paths.push(pathKey(operands.at(-1), cwd));
      if (name === "mv") paths.push(...operands.slice(0, -1).map(word => pathKey(word, cwd)));
    }
  } else if (name === "rm" || name === "touch" || name === "truncate") paths.push(...rest.filter(word => !word.startsWith("-")).map(word => pathKey(word, cwd)));
  else if (name === "git" && (rest[0] === "checkout" || rest[0] === "restore")) {
    const operands = rest.includes("--") ? rest.slice(rest.indexOf("--") + 1) : rest.slice(1).filter(word => !word.startsWith("-"));
    if (operands.length === 1 && operands[0] === ".") return [ANY_FILE];
    if (!operands.length) return [ANY_FILE];
    paths.push(...operands.map(word => pathKey(word, cwd)));
  } else if (name === "git" && rest[0] === "stash" && ["push", "pop", "apply", "save", "-q"].includes(rest[1] ?? "")) return [ANY_FILE];
  else if (name === "git" && rest[0] === "stash" && rest.length === 1) return [ANY_FILE];
  else if ((name === "git" && GIT_ANY_SUBCOMMANDS.includes(rest[0] ?? "")) || name === "patch") {
    if (!rest.some(word => ["--stat", "--numstat", "--summary", "--check", "--dry-run"].includes(word))) return [ANY_FILE];
  }
  // Redirections: `>` and `>>` with a target that is not a descriptor.
  for (let i = 0; i < words.length - 1; i++)
    if ((words[i] === ">" || words[i] === ">>") && !words[i + 1]!.startsWith("&")) paths.push(pathKey(words[i + 1], cwd));
  return paths.filter((path): path is string => path !== undefined && path !== "/dev/null");
}

/** What one tool call viewed, wrote, ran or searched. */
export function footprints(tool: string, input: Record<string, unknown> | undefined, cwd?: string): Footprint[] {
  if (!input) return [];
  const name = tool.toLowerCase();
  const target = input.filePath ?? input.path ?? input.file_path;
  if (VIEW_TOOLS.has(name)) {
    const key = pathKey(target, cwd);
    return key ? [{ kind: "view", key, lines: readLines(input) }] : [];
  }
  if (WRITE_TOOLS.has(name)) return [{ kind: "write", key: pathKey(target, cwd) ?? ANY_FILE }];
  if (SEARCH_TOOLS.has(name)) return [{ kind: "search", key: name + JSON.stringify(input) }];
  if (RUN_TOOLS.has(name)) {
    const command = typeof input.command === "string" && input.command.trim() ? input.command : commandOf(tool, input)?.command;
    if (typeof command !== "string" || !command.trim()) return [];
    let workdir = pathKey(input.workdir ?? input.cwd, cwd) ?? cwd;
    const cd = CD_PREFIX.exec(command);
    const body = cd ? command.slice(cd[0].length) : command;
    if (cd) workdir = pathKey(cd[1], workdir) ?? workdir;
    const found: Footprint[] = [{ kind: "run", key: `${workdir ?? ""}$ ${normalise(body)}` }];
    const parts = simpleCommands(body);
    if (!parts) return found;
    if (parts.length === 1) {
      const view = viewOf(parts[0]!, workdir);
      if (view) found.push(view);
    }
    // "git stash && run the tests && git stash pop" checks the old code and puts the change back: no file ends up different.
    const stashes = parts.filter(words => words[0] === "git" && words[1] === "stash");
    const restored = stashes.length > 0 && ["pop", "apply"].includes(stashes.at(-1)![2] ?? "")
      && stashes.some(words => [undefined, "push", "save", "-q"].includes(words[2]));
    for (const words of parts) {
      if (restored && words[0] === "git" && words[1] === "stash") continue;
      for (const path of writesOf(words, workdir)) found.push({ kind: "write", key: path });
    }
    return found;
  }
  return [];
}

/* ─── Relations ─────────────────────────────────────────────────────── */

/** One observed call: what the relevance compaction has for a tool unit. */
export interface ObservedCall {
  id: string;
  tool: string;
  input: Record<string, unknown> | undefined;
}

interface ObservedOutput {
  call: ObservedCall;
  prints: Footprint[];
  /** Line ranges of this output's file that later views have shown again. */
  covered: Array<{ start: number; end: number }>;
}

/** Whether `spans` together cover `target` from its start to its end. */
function covers(spans: ReadonlyArray<{ start: number; end: number }>, target: { start: number; end: number }): boolean {
  let reach = target.start;
  for (const { start, end } of [...spans].sort((a, b) => a.start - b.start)) {
    if (start > reach) break;
    if (end >= target.end) return true;
    reach = Math.max(reach, end + 1);
  }
  return false;
}

/** Why `later` supersedes `earlier`, or undefined; a covering view is recorded on `earlier` even when it does not cover yet. */
function supersedes(later: readonly Footprint[], earlier: ObservedOutput): string | undefined {
  for (const old of earlier.prints) {
    for (const fresh of later) {
      if (old.kind !== fresh.kind || old.key !== fresh.key) continue;
      if (old.kind === "run") return "the same command was run again";
      if (old.kind === "search") return "the same search was run again";
      if (old.kind === "view" && old.lines && fresh.lines) {
        // Several narrower views can between them show all an earlier one did.
        earlier.covered.push(fresh.lines);
        if (covers(earlier.covered, old.lines)) return `${old.key} was viewed again over the same lines`;
      }
    }
  }
  return undefined;
}

/** The relation that matters for `older`: superseded before stale, whatever their order. */
function statusOf(recorded: ReadonlyMap<string, Relation>, older: string): Relation | undefined {
  return recorded.get(`${older}:superseded`) ?? recorded.get(`${older}:stale`);
}

/**
 * The supersession relations among the calls, in span order. Calls whose input is not known (read back from an earlier
 * compaction) take no part: what they viewed or ran cannot be seen. Each older call carries at most one relation:
 * superseded before stale, whichever came first.
 */
export function relations(calls: readonly ObservedCall[], cwd?: string): Relation[] {
  const outputs: ObservedOutput[] = [];
  const found: Relation[] = [];
  const recorded = new Map<string, Relation>();
  const keep = (relation: Relation) => { found.push(relation); recorded.set(`${relation.older}:${relation.kind}`, relation); };
  for (const call of calls) {
    const prints = footprints(call.tool, call.input, cwd);
    const writes = new Set(prints.filter(print => print.kind === "write").map(print => print.key));
    for (const earlier of outputs) {
      const status = statusOf(recorded, earlier.call.id);
      if (earlier.call.id === call.id || status?.kind === "superseded") continue;
      const reason = supersedes(prints, earlier);
      if (reason) { keep({ older: earlier.call.id, newer: call.id, kind: "superseded", reason }); continue; }
      if (writes.size && status === undefined) {
        for (const view of earlier.prints) {
          if (view.kind === "view" && (writes.has(view.key) || writes.has(ANY_FILE))) {
            keep({ older: earlier.call.id, newer: call.id, kind: "stale", reason: `${view.key} was changed after this output` });
            break;
          }
        }
      }
    }
    if (prints.some(print => print.kind === "view" || print.kind === "run" || print.kind === "search")) outputs.push({ call, prints, covered: [] });
  }
  return found;
}

/** The note a pre-pruned unit's left-out line carries, preferring superseded over stale whatever their order. */
export function supersedeNote(list: readonly Relation[]): string | undefined {
  return (list.find(relation => relation.kind === "superseded") ?? list[0])?.reason;
}
