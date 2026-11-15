// lib/providers.ts — dialing free models through the trial gateway.
//
// What this file owns:
//   * the free-model table (built from the trial brief at runtime, never
//     hard-coded) and the rating-first ranking that mirrors the house
//     scorer's best-rating rule: rating desc, then context desc, then id asc,
//     keeping only models that meet the prompt's requirements (json / tools).
//   * the gateway call itself (Chat Completions only, on OPENAI_BASE_URL),
//     normalising every upstream answer into one wire-independent shape and
//     backfilling a WHOLE usage object before anything translates it —
//     input_tokens defaults to 0, output_tokens is forced to at least 1, so
//     no translator downstream can ever emit a null usage.
//   * the failover walk: per prompt, cycle the 3 best eligible models, up to
//     6 attempts. A 402 demotes that model for the rest of the trial (its
//     budget is gone, retrying it is a wasted call); a 429 / 5xx / other
//     refusal only moves on to the next candidate, and the same model comes
//     back around on the second pass — a transient first-call refusal (the
//     mock's scripted 429/500s, a real rate-limit burst) is recovered by the
//     retry instead of burning the prompt. Every refusal is recorded in
//     order for the x-makefun-upstream-failures trail.
//
// Determinism: every decision here is a function of the brief, the request
// and the gateway answers seen so far. No wall clock, no randomness.

export interface FreeModel {
  /** Dialable id, e.g. "openrouter/thinkingmachines/inkling-2-small:free". */
  id: string;
  /** House quality rating; higher is stronger. */
  rating: number;
  /** Context window in tokens. */
  context: number;
  /** Supports JSON mode. */
  json: boolean;
  /** Supports tools. */
  tools: boolean;
}

export interface PromptNeeds {
  json: boolean;
  tools: boolean;
}

export interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

export interface ToolCall {
  id: string;
  name: string;
  /** Raw arguments JSON as the provider sent it. */
  args: string;
}

/** One upstream answer, normalised and usage-backfilled, wire-independent. */
export interface UpstreamAnswer {
  /** Concatenated text content (may be empty when only tools were called). */
  text: string;
  toolCalls: ToolCall[];
  /** Upstream finish_reason, verbatim. */
  finish: string;
  /** Whole input usage; 0 when the provider reported none. */
  inputTokens: number;
  /** Whole output usage; always >= 1. */
  outputTokens: number;
}

export type CallResult =
  | { ok: true; answer: UpstreamAnswer }
  | { ok: false; status: number; code: string };

export interface GatewayConfig {
  baseUrl: string;
  apiKey: string;
}

/** The scorer's own requirement gate: json / tools only. */
export function meets(model: FreeModel, needs: PromptNeeds): boolean {
  if (needs.json && !model.json) return false;
  if (needs.tools && !model.tools) return false;
  return true;
}

/** Eligible models, best first: rating desc, context desc, id asc. */
export function ranked(models: FreeModel[], needs: PromptNeeds): FreeModel[] {
  return models
    .filter((m) => meets(m, needs))
    .sort(
      (a, b) =>
        b.rating - a.rating ||
        b.context - a.context ||
        (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
    );
}

function numOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Whole usage, computed from the provider's own numbers the moment the
 * answer arrives. This runs BEFORE any wire translation: a translator that
 * received a half-filled usage could emit a null one, and a null output
 * count fails the prompt. When the provider omits usage entirely, the
 * output count is estimated from the answer length (a token is roughly
 * four characters) so the field stays present and >= 1.
 */
function wholeUsage(usage: unknown, text: string): {
  inputTokens: number;
  outputTokens: number;
} {
  const u = (usage !== null && typeof usage === "object"
    ? (usage as Record<string, unknown>)
    : {}) as Record<string, unknown>;
  const input = Math.max(0, numOr(u.prompt_tokens, 0));
  let output = numOr(u.completion_tokens, 0);
  if (!(output >= 1)) output = Math.max(1, Math.ceil(text.length / 4));
  return { inputTokens: input, outputTokens: output };
}

/** Join a Chat Completions message content (string or text parts) to text. */
function flattenContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let out = "";
  for (const part of content) {
    if (
      part !== null &&
      typeof part === "object" &&
      (part as Record<string, unknown>).type === "text" &&
      typeof (part as Record<string, unknown>).text === "string"
    ) {
      out += (part as Record<string, unknown>).text as string;
    }
  }
  return out;
}

function normalizeToolCalls(raw: unknown): ToolCall[] {
  if (!Array.isArray(raw)) return [];
  const calls: ToolCall[] = [];
  for (const entry of raw) {
    if (entry === null || typeof entry !== "object") continue;
    const fn = (entry as Record<string, unknown>).function as
      | Record<string, unknown>
      | undefined;
    if (!fn || typeof fn.name !== "string") continue;
    const args = fn.arguments;
    calls.push({
      id:
        typeof (entry as Record<string, unknown>).id === "string"
          ? ((entry as Record<string, unknown>).id as string)
          : "",
      name: fn.name,
      args:
        typeof args === "string"
          ? args
          : args === undefined
            ? "{}"
            : JSON.stringify(args),
    });
  }
  return calls;
}

async function refusalCode(res: Response, status: number): Promise<string> {
  try {
    const data = (await res.clone().json()) as Record<string, unknown>;
    const err = data.error as Record<string, unknown> | undefined;
    if (err) {
      if (typeof err.code === "string" && err.code) return err.code;
      if (typeof err.type === "string" && err.type) return err.type;
      if (typeof err.message === "string" && err.message)
        return err.message.slice(0, 80);
    }
    if (typeof data.code === "string" && data.code) return data.code;
  } catch {
    // Body is not JSON; fall through to the status-based code.
  }
  return `http_${status}`;
}

function readNonStream(body: unknown): UpstreamAnswer {
  const data = (body !== null && typeof body === "object" ? body : {}) as Record<
    string,
    unknown
  >;
  const choice = (Array.isArray(data.choices) ? data.choices[0] : undefined) as
    | Record<string, unknown>
    | undefined;
  const message = (
    choice !== undefined &&
    choice.message !== null &&
    typeof choice.message === "object"
      ? (choice.message as Record<string, unknown>)
      : {}
  ) as Record<string, unknown>;
  const text = flattenContent(message.content);
  const toolCalls = normalizeToolCalls(message.tool_calls);
  const finish =
    typeof choice?.finish_reason === "string" ? choice.finish_reason : "stop";
  return { text, toolCalls, finish, ...wholeUsage(data.usage, text) };
}

function readStreamed(raw: string): UpstreamAnswer {
  let text = "";
  const deltas = new Map<number, ToolCall>();
  let finish = "stop";
  let usage: unknown = null;
  for (const line of raw.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;
    const payload = line.slice("data:".length).trim();
    if (payload === "" || payload === "[DONE]") continue;
    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      continue;
    }
    const choice = (Array.isArray(chunk.choices) ? chunk.choices[0] : undefined) as
      | Record<string, unknown>
      | undefined;
    const delta = (
      choice !== undefined &&
      choice.delta !== null &&
      typeof choice.delta === "object"
        ? (choice.delta as Record<string, unknown>)
        : {}
    ) as Record<string, unknown>;
    if (typeof delta.content === "string") text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls as Record<string, unknown>[]) {
        const index =
          typeof tc.index === "number" ? (tc.index as number) : 0;
        const cur = deltas.get(index) ?? { id: "", name: "", args: "" };
        if (typeof tc.id === "string") cur.id = tc.id;
        const fn = tc.function as Record<string, unknown> | undefined;
        if (fn) {
          if (typeof fn.name === "string") cur.name = fn.name;
          if (typeof fn.arguments === "string") cur.args += fn.arguments;
        }
        deltas.set(index, cur);
      }
    }
    if (typeof choice?.finish_reason === "string")
      finish = choice.finish_reason;
    if (chunk.usage !== undefined) usage = chunk.usage;
  }
  const toolCalls = [...deltas.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, call]) => call);
  return { text, toolCalls, finish, ...wholeUsage(usage, text) };
}

/**
 * Dial one model on the gateway's Chat Completions route. The request body
 * is already a Chat Completions body (the caller translated it). A non-2xx
 * answer is a refusal, never an exception: the failover loop decides what
 * to do with it. A transport failure is reported as a 502 refusal.
 */
export async function callModel(
  cfg: GatewayConfig,
  modelId: string,
  body: Record<string, unknown>,
): Promise<CallResult> {
  const stream = body.stream === true;
  let res: Response;
  try {
    res = await fetch(`${cfg.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${cfg.apiKey}`,
      },
      body: JSON.stringify(body),
    });
  } catch {
    return { ok: false, status: 502, code: "fetch_failed" };
  }
  if (!res.ok) {
    return { ok: false, status: res.status, code: await refusalCode(res, res.status) };
  }
  const answer = stream
    ? readStreamed(await res.text())
    : readNonStream(await res.json());
  return { ok: true, answer };
}

export type ServeOutcome =
  | {
      ok: true;
      model: string;
      answer: UpstreamAnswer;
      trail: TrailRow[];
    }
  | { ok: false; trail: TrailRow[] };

/**
 * Trial-scoped router + failover. `dial` performs one gateway call for a
 * model id; this class decides which model to try, in which order, and when
 * a refusal is permanent (402) versus transient (anything else).
 *
 * Walk: eligible models best-first, top 3 as the retry set, up to 6
 * attempts. A 402 removes the model from the set for the rest of the trial
 * and the set refills from the next-best eligible model. Any other refusal
 * only advances to the next candidate; the cycle wraps, so the same model
 * is retried on the second pass once its transient refusal has cleared.
 */
export class UpstreamPool {
  private demoted = new Set<string>();
  private readonly models: FreeModel[];

  constructor(models: FreeModel[]) {
    this.models = models;
  }

  async serve(
    needs: PromptNeeds,
    dial: (modelId: string) => Promise<CallResult>,
  ): Promise<ServeOutcome> {
    const trail: TrailRow[] = [];
    let attempts = 0;
    while (attempts < 6) {
      const cycle = ranked(this.models, needs)
        .filter((m) => !this.demoted.has(m.id))
        .slice(0, 3);
      if (cycle.length === 0) break;
      const model = cycle[attempts % cycle.length];
      attempts += 1;
      const res = await dial(model.id);
      if (res.ok) {
        return { ok: true, model: model.id, answer: res.answer, trail };
      }
      trail.push({ status: res.status, code: res.code, model: model.id });
      if (res.status === 402) this.demoted.add(model.id);
    }
    return { ok: false, trail };
  }
}
