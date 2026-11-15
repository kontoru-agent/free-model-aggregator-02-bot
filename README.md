# free-model-aggregator-02 — live free-model aggregator

One server that answers both completion wires (OpenAI Chat Completions and
Anthropic Messages, JSON and streaming) with REAL completions from the
fourteen free models in the trial brief, dialled through the grader's
gateway on Chat Completions. It never answers from itself.

## Layout

- `main.ts` — entrypoint: reads the brief, builds the sealed catalog,
  serves `GET /v1/models` and both completion routes, registers with the
  env, polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when it says done.
  `node main.ts --list` prints the catalog and exits.
- `lib/providers.ts` — the model table, rating-first ranking, the gateway
  call with whole-usage backfill, and the failover walk.
- `lib/wire.ts` — translation between the client wires and Chat
  Completions, model-field rewrite, tool filtering, stop-reason mapping.

## How it picks a model

Rating-first, read off the request, never hard-coded per prompt: a prompt
asking for JSON mode gets the best JSON-capable model, anything else gets
the best model overall; ties break by context length, then id. That is the
same order the house scorer uses for its best rating, so on a clean trial
the served model is the top one for every prompt.

## How it survives refusals

Per prompt it walks the 3 best eligible models, up to 6 attempts. A 402
demotes that model for the rest of the trial (its budget is gone); a 429
or 5xx only moves to the next candidate, and the cycle wraps so the same
model is retried on a second pass once a transient refusal has cleared.
Every refusal is reported in order on the
`x-makefun-upstream-failures` header (at most 8 rows), which is the only
way the failover column can see a refusal that was recovered from.

## Wire notes

- The `model` field on every answer and every streamed chunk is always the
  id actually dialled. Providers echo their bare slug; the house scores the
  model the answer names, so the dialled id is written back.
- Usage is backfilled the moment the upstream answer arrives, before any
  translation: input defaults to 0, output is forced to at least 1.
- Streaming is buffered, then re-emitted in canonical order, so the stream
  shape never depends on how the provider chunked its own stream.
- Tool calls are only carried through when the client declared that tool
  name for the prompt.
- No wall clock, no unseeded randomness: ids come from a counter,
  timestamps from `MAKEFUN_TRIAL_NOW`, so a replay leaves the same export.

## Grade locally

From the kit directory, with the mock gateway on 127.0.0.1:8931:

```
node cli/makefun.mjs grade-local --kit . --bot <this directory> \
  --driver process --worktree --instances 3 \
  --gateway http://127.0.0.1:8931 --save <out dir>
```

`MOCK_FAIL_ALL=1` on the mock gateway re-runs trial 0 under a total
outage (every model's first call 500s).
