// main.ts — the best live free-model aggregator (free-model-aggregator-02).
//
// One long-running server. It listens on 0.0.0.0 at a free port, registers
// that port and its catalog with the env, answers both completion wires
// (OpenAI Chat Completions, Anthropic Messages — JSON and streaming), and
// answers every prompt with a REAL completion from a free model dialled
// through the trial gateway. It never answers from itself, and it never
// decides from the wall clock: ids come from a counter, timestamps from
// MAKEFUN_TRIAL_NOW, and every routing choice comes from the brief, the
// request and the gateway answers. When GET $MAKEFUN_ENV_URL/work says
// done, it exits 0.
//
// Run:  node main.ts [--list]
//   --list prints the catalog document (byte for byte the one registered)
//   and exits 0.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import { callModel, UpstreamPool } from "./lib/providers.ts";
import type { FreeModel, GatewayConfig } from "./lib/providers.ts";
import {
  anthropicResponse,
  anthropicStream,
  declaredToolNames,
  errorBody,
  needsOf,
  openaiResponse,
  openaiStream,
  toChatBody,
  trailHeader,
} from "./lib/wire.ts";
import type { Wire } from "./lib/wire.ts";

type AnyRecord = Record<string, unknown>;

const CATALOG_PROVIDER = "openrouter";
const CATALOG_BASE_URL = "https://openrouter.ai/api/v1";
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/* ------------------------------------------------------------------ */
/* Brief                                                               */
/* ------------------------------------------------------------------ */

interface BriefModel {
  id: string;
  display_name?: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  json: boolean;
}

/**
 * Read the free-model table out of the trial brief (MAKEFUN_PROMPT_FILE).
 * The brief carries the table as one JSON line; the scan takes the first
 * line that parses to an object holding a `models` array. No brief means
 * no models (and --list then prints an empty catalog).
 */
function readBriefModels(): BriefModel[] {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return [];
  let text: string;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return [];
  }
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    try {
      const doc = JSON.parse(trimmed) as AnyRecord;
      if (Array.isArray(doc.models)) return doc.models as BriefModel[];
    } catch {
      // Not a JSON blob line; keep scanning.
    }
  }
  return [];
}

function toFreeModels(models: BriefModel[]): FreeModel[] {
  return models.map((m) => ({
    id: m.id,
    rating: m.rating_milli,
    context: m.context_length,
    json: Boolean(m.json),
    tools: Boolean(m.tools),
  }));
}

/**
 * The catalog document, exactly the starter's shape: one openrouter
 * provider, the fourteen free models from the brief, each once. The same
 * serialised string is printed by --list and sent at register time.
 */
function buildCatalog(models: BriefModel[]): AnyRecord {
  return {
    providers: [
      {
        id: CATALOG_PROVIDER,
        base_url: CATALOG_BASE_URL,
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({
          id: m.id,
          free: true,
          context_length: m.context_length ?? null,
        })),
      },
    ],
  };
}

function emptyCatalog(): AnyRecord {
  return { providers: [] };
}

/* ------------------------------------------------------------------ */
/* Trial time (never the wall clock)                                   */
/* ------------------------------------------------------------------ */

function trialNowIso(): string {
  return process.env.MAKEFUN_TRIAL_NOW ?? "2026-11-15T03:30:00.000Z";
}

function trialCreatedSeconds(): number {
  const raw = process.env.MAKEFUN_TRIAL_NOW;
  if (raw) {
    const ms = Date.parse(raw);
    if (Number.isFinite(ms)) return Math.floor(ms / 1000);
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* HTTP helpers                                                        */
/* ------------------------------------------------------------------ */

function sendJson(
  res: ServerResponse,
  status: number,
  body: AnyRecord,
  trail?: string,
): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(text)),
  };
  if (trail !== undefined) headers["x-makefun-upstream-failures"] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: Wire,
  status: number,
  message: string,
  code: string,
  trail?: string,
): void {
  sendJson(res, status, errorBody(wire, message, code), trail);
}

function sendSse(res: ServerResponse, blocks: string[], trail?: string): void {
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  };
  if (trail !== undefined) headers["x-makefun-upstream-failures"] = trail;
  res.writeHead(200, headers);
  for (const block of blocks) res.write(`${block}\n\n`);
  res.end();
}

function readBody(req: IncomingMessage, res: ServerResponse, wire: Wire): Promise<string | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let overflow = false;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        overflow = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (overflow) {
        sendError(res, wire, 413, "request body too large", "request_too_large");
        resolve(null);
        return;
      }
      resolve(Buffer.concat(chunks).toString("utf8"));
    });
    req.on("error", () => {
      sendError(res, wire, 400, "could not read request body", "invalid_request");
      resolve(null);
    });
  });
}

/* ------------------------------------------------------------------ */
/* Server context                                                      */
/* ------------------------------------------------------------------ */

interface Context {
  pool: UpstreamPool;
  gateway: GatewayConfig;
  models: BriefModel[];
  created: number;
  trialIso: string;
  sequence: number;
}

function nextId(ctx: Context, prefix: string): string {
  ctx.sequence += 1;
  return `${prefix}-agg-${ctx.sequence}`;
}

function handleModels(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: Context,
): void {
  const ids = ctx.models.map((m) => m.id);
  const apiKeyHeader = req.headers["x-api-key"];
  if (typeof apiKeyHeader === "string" && apiKeyHeader !== "") {
    // Anthropic shape for an x-api-key request.
    sendJson(res, 200, {
      data: ids.map((id, i) => ({
        id,
        type: "model",
        display_name: ctx.models[i].display_name ?? id,
        created_at: ctx.trialIso,
      })),
      has_more: false,
      first_id: ids[0] ?? null,
      last_id: ids[ids.length - 1] ?? null,
    });
    return;
  }
  // OpenAI shape for an Authorization: Bearer request (and the default).
  sendJson(res, 200, {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: ctx.created,
      owned_by: "openrouter",
    })),
  });
}

async function handlePrompt(
  wire: Wire,
  req: IncomingMessage,
  res: ServerResponse,
  ctx: Context,
): Promise<void> {
  const raw = await readBody(req, res, wire);
  if (raw === null) return;
  let body: AnyRecord;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      sendError(res, wire, 400, "request body is not an object", "invalid_request");
      return;
    }
    body = parsed as AnyRecord;
  } catch {
    sendError(res, wire, 400, "request body is not JSON", "invalid_json");
    return;
  }
  const stream = body.stream === true;
  const needs = needsOf(wire, body);
  const declared = declaredToolNames(wire, body);
  const outcome = await ctx.pool.serve(needs, (modelId) =>
    callModel(ctx.gateway, modelId, toChatBody(wire, body, modelId)),
  );
  const trail = trailHeader(outcome.trail);
  if (!outcome.ok) {
    // No model answered after the full failover walk. The prompt scores
    // nothing; the 500 plus the trail (the cassette holds the refusals
    // behind it) is the honest record of that.
    sendError(
      res,
      wire,
      500,
      "no free model answered this prompt",
      "upstream_unavailable",
      trail,
    );
    return;
  }
  if (wire === "openai") {
    const id = nextId(ctx, "chatcmpl");
    if (stream) {
      sendSse(
        res,
        openaiStream(outcome.answer, outcome.model, id, ctx.created, declared),
        trail,
      );
      return;
    }
    sendJson(
      res,
      200,
      openaiResponse(outcome.answer, outcome.model, id, ctx.created, declared),
      trail,
    );
    return;
  }
  const id = nextId(ctx, "msg");
  if (stream) {
    sendSse(res, anthropicStream(outcome.answer, outcome.model, id, declared), trail);
    return;
  }
  sendJson(res, 200, anthropicResponse(outcome.answer, outcome.model, id, declared), trail);
}

/* ------------------------------------------------------------------ */
/* Trial lifecycle                                                     */
/* ------------------------------------------------------------------ */

async function registerWithEnv(
  envUrl: string,
  port: number,
  catalog: AnyRecord,
): Promise<boolean> {
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
      });
      if (res.ok) return true;
    } catch {
      // Transient; retry below.
    }
    await sleep(500);
  }
  return false;
}

async function waitForDone(envUrl: string): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`);
      if (res.ok) {
        const work = (await res.json()) as AnyRecord;
        if (work.done === true) return;
      }
    } catch {
      // Transient; poll again below.
    }
    await sleep(250);
  }
}

async function main(): Promise<void> {
  const brief = readBriefModels();
  const catalog = brief.length > 0 ? buildCatalog(brief) : emptyCatalog();
  const catalogText = JSON.stringify(catalog);
  if (process.argv.includes("--list")) {
    process.stdout.write(`${catalogText}\n`);
    process.exit(0);
  }
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  if (envUrl === "") {
    console.error("MAKEFUN_ENV_URL is not set; nothing to register with.");
    process.exit(1);
  }
  const baseUrl = (process.env.OPENAI_BASE_URL ?? "").replace(/\/$/, "");
  if (baseUrl === "") {
    console.error("OPENAI_BASE_URL is not set; no gateway to dial.");
    process.exit(1);
  }
  const ctx: Context = {
    pool: new UpstreamPool(toFreeModels(brief)),
    gateway: {
      baseUrl,
      apiKey: process.env.OPENAI_API_KEY ?? "",
    },
    models: brief,
    created: trialCreatedSeconds(),
    trialIso: trialNowIso(),
    sequence: 0,
  };
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://entry");
    const path = url.pathname;
    void (async () => {
      try {
        if (req.method === "GET" && path === "/v1/models") {
          handleModels(req, res, ctx);
        } else if (req.method === "POST" && path === "/v1/chat/completions") {
          await handlePrompt("openai", req, res, ctx);
        } else if (req.method === "POST" && path === "/v1/messages") {
          await handlePrompt("anthropic", req, res, ctx);
        } else {
          sendJson(res, 404, {
            error: { message: "not found", type: "not_found", code: "not_found" },
          });
        }
      } catch (err) {
        if (!res.headersSent) {
          sendJson(res, 500, {
            error: {
              message: err instanceof Error ? err.message : "internal error",
              type: "server_error",
              code: "internal_error",
            },
          });
        } else {
          res.end();
        }
      }
    })();
  });
  await new Promise<void>((resolve) =>
    server.listen(0, "0.0.0.0", () => resolve()),
  );
  const address = server.address();
  const port =
    address !== null && typeof address === "object" ? address.port : 0;
  const registered = await registerWithEnv(envUrl, port, catalog);
  if (!registered) {
    console.error("could not register with the env; exiting.");
    process.exit(1);
  }
  await waitForDone(envUrl);
  server.close();
  process.exit(0);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
