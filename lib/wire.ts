// lib/wire.ts — translation between the two client wires and the single
// upstream wire (OpenAI Chat Completions).
//
// Every function here receives an UpstreamAnswer whose usage was already
// backfilled by providers.ts, so usage is never null at this point: the
// translators only re-shape it, they never invent it.
//
// Rules enforced here:
//   * the `model` field on every answer and every streamed chunk is always
//     the id that was actually dialled. The provider echoes its own slug
//     (bare, without the gateway's openrouter/ prefix); the house scores
//     the model the answer names, so the dialled id is written back.
//   * a tool call is only carried through when its name is one the client
//     declared for that prompt; an undeclared tool name in an answer fails
//     the prompt's tools check, so it is dropped, never echoed.
//   * stop reasons are mapped per wire from the upstream finish_reason.
//   * streaming is buffered then re-emitted: the whole upstream stream is
//     read first, then the client wire is emitted in its canonical order
//     (role chunk, content deltas, final chunk with finish + usage, [DONE]
//     on OpenAI; message_start, content blocks, message_delta, message_stop
//     on Anthropic). Order is what the wire checks judge; timing is not.

import type { PromptNeeds, ToolCall, UpstreamAnswer } from "./providers.ts";

export type Wire = "openai" | "anthropic";

type AnyRecord = Record<string, unknown>;

function isRecord(value: unknown): value is AnyRecord {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** What the prompt needs, read off the request itself (never a manifest). */
export function needsOf(wire: Wire, body: AnyRecord): PromptNeeds {
  const json =
    wire === "openai" &&
    isRecord(body.response_format) &&
    body.response_format.type === "json_object";
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  return { json, tools };
}

/** Tool names the client declared for this prompt, for filtering answers. */
export function declaredToolNames(wire: Wire, body: AnyRecord): Set<string> {
  const names = new Set<string>();
  if (!Array.isArray(body.tools)) return names;
  for (const tool of body.tools as AnyRecord[]) {
    if (!isRecord(tool)) continue;
    if (wire === "openai") {
      const fn = tool.function;
      if (isRecord(fn) && typeof fn.name === "string") names.add(fn.name);
    } else if (typeof tool.name === "string") {
      names.add(tool.name);
    }
  }
  return names;
}

function capTokens(value: unknown): number {
  const n = typeof value === "number" && Number.isFinite(value) ? value : 1024;
  return Math.min(8192, Math.max(1, Math.floor(n)));
}

function openaiCap(body: AnyRecord): number {
  return capTokens(body.max_tokens ?? body.max_completion_tokens);
}

function systemText(system: unknown): string {
  if (typeof system === "string") return system;
  if (Array.isArray(system)) {
    let out = "";
    for (const part of system as AnyRecord[]) {
      if (isRecord(part) && part.type === "text" && typeof part.text === "string")
        out += part.text;
    }
    return out;
  }
  return "";
}

function anthropicPartToChat(part: AnyRecord): AnyRecord | null {
  if (part.type === "text" && typeof part.text === "string") {
    return { type: "text", text: part.text };
  }
  if (part.type === "image" && isRecord(part.source)) {
    const src = part.source;
    if (src.type === "base64" && typeof src.data === "string") {
      const media = typeof src.media_type === "string" ? src.media_type : "image/png";
      return {
        type: "image_url",
        image_url: { url: `data:${media};base64,${src.data}` },
      };
    }
  }
  return null;
}

/** One Anthropic message -> one or two Chat Completions messages. */
function anthropicMessageToChat(message: AnyRecord): AnyRecord[] {
  const role = message.role === "assistant" ? "assistant" : "user";
  const content = message.content;
  if (typeof content === "string") return [{ role, content }];
  if (!Array.isArray(content)) return [{ role, content: "" }];
  const texts: string[] = [];
  const parts: AnyRecord[] = [];
  const toolUses: AnyRecord[] = [];
  const toolResults: AnyRecord[] = [];
  for (const part of content as AnyRecord[]) {
    if (!isRecord(part)) continue;
    if (part.type === "text" && typeof part.text === "string") {
      texts.push(part.text);
      const mapped = anthropicPartToChat(part);
      if (mapped) parts.push(mapped);
    } else if (part.type === "image") {
      const mapped = anthropicPartToChat(part);
      if (mapped) parts.push(mapped);
    } else if (part.type === "tool_use") {
      toolUses.push(part);
    } else if (part.type === "tool_result") {
      toolResults.push(part);
    }
  }
  const out: AnyRecord[] = [];
  if (toolResults.length > 0) {
    if (texts.length > 0) out.push({ role: "user", content: texts.join("") });
    for (const result of toolResults) {
      const resultContent =
        typeof result.content === "string"
          ? result.content
          : JSON.stringify(result.content ?? "");
      out.push({
        role: "tool",
        tool_call_id: typeof result.tool_use_id === "string" ? result.tool_use_id : "",
        content: resultContent,
      });
    }
    return out;
  }
  if (toolUses.length > 0) {
    const messageOut: AnyRecord = {
      role: "assistant",
      content: parts.length === 1 && parts[0].type === "text" ? parts[0].text : parts,
    };
    messageOut.tool_calls = toolUses.map((use) => ({
      id: typeof use.id === "string" ? use.id : "",
      type: "function",
      function: {
        name: typeof use.name === "string" ? use.name : "",
        arguments: JSON.stringify(use.input ?? {}),
      },
    }));
    if (typeof messageOut.content === "string" && messageOut.content === "")
      messageOut.content = null;
    out.push(messageOut);
    return out;
  }
  return [
    {
      role,
      content: parts.length === 1 && parts[0].type === "text" ? parts[0].text : parts,
    },
  ];
}

function anthropicChoiceToOpenAI(choice: unknown): unknown {
  if (!isRecord(choice)) return "auto";
  if (choice.type === "auto") return "auto";
  if (choice.type === "any") return "required";
  if (choice.type === "tool" && typeof choice.name === "string") {
    return { type: "function", function: { name: choice.name } };
  }
  if (choice.type === "none") return "none";
  return "auto";
}

/**
 * Build the Chat Completions body for one dialled model from a client
 * request on either wire. Messages, system prompt, tools, caps and
 * sampling pass through; the model is always the dialled id.
 */
export function toChatBody(
  wire: Wire,
  body: AnyRecord,
  modelId: string,
): AnyRecord {
  const out: AnyRecord = { model: modelId };
  if (wire === "openai") {
    out.messages = Array.isArray(body.messages) ? body.messages : [];
    out.max_tokens = openaiCap(body);
    for (const key of ["temperature", "top_p", "stop", "tools", "tool_choice", "response_format", "seed"] as const) {
      if (body[key] !== undefined) out[key] = body[key];
    }
  } else {
    const messages: AnyRecord[] = [];
    if (body.system !== undefined) {
      messages.push({ role: "system", content: systemText(body.system) });
    }
    for (const message of Array.isArray(body.messages) ? body.messages : []) {
      if (isRecord(message)) messages.push(...anthropicMessageToChat(message));
    }
    out.messages = messages;
    out.max_tokens = capTokens(body.max_tokens);
    for (const key of ["temperature", "top_p"] as const) {
      if (typeof body[key] === "number") out[key] = body[key];
    }
    if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) {
      out.stop = body.stop_sequences;
    }
    if (Array.isArray(body.tools) && body.tools.length > 0) {
      out.tools = (body.tools as AnyRecord[]).map((tool) => ({
        type: "function",
        function: {
          name: typeof tool.name === "string" ? tool.name : "",
          description: typeof tool.description === "string" ? tool.description : "",
          parameters: isRecord(tool.input_schema)
            ? tool.input_schema
            : { type: "object", properties: {} },
        },
      }));
      if (body.tool_choice !== undefined) {
        out.tool_choice = anthropicChoiceToOpenAI(body.tool_choice);
      }
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Tool calls the answer may carry: declared names only, in order. */
function allowedCalls(
  answer: UpstreamAnswer,
  declared: Set<string>,
): ToolCall[] {
  return answer.toolCalls.filter((call) => declared.has(call.name));
}

function openaiStop(finish: string, hasToolCalls: boolean): string {
  if (hasToolCalls && (finish === "tool_calls" || finish === "function_call"))
    return "tool_calls";
  switch (finish) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
      return "tool_calls";
    case "function_call":
      return "function_call";
    case "content_filter":
      return "content_filter";
    default:
      return "stop";
  }
}

function anthropicStop(finish: string, hasToolCalls: boolean): string {
  if (hasToolCalls) return "tool_use";
  switch (finish) {
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "refusal";
    default:
      return "end_turn";
  }
}

function openaiUsage(answer: UpstreamAnswer): AnyRecord {
  return {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
}

function parseToolArgs(args: string): unknown {
  if (args === "") return {};
  try {
    return JSON.parse(args);
  } catch {
    return {};
  }
}

function toolCallId(call: ToolCall, fallback: string): string {
  return call.id !== "" ? call.id : fallback;
}

/** Non-streamed OpenAI Chat Completions answer. */
export function openaiResponse(
  answer: UpstreamAnswer,
  modelId: string,
  id: string,
  created: number,
  declared: Set<string>,
): AnyRecord {
  const calls = allowedCalls(answer, declared);
  const message: AnyRecord = { role: "assistant", content: answer.text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((call, i) => ({
      id: toolCallId(call, `${id}-${i}`),
      type: "function",
      function: { name: call.name, arguments: call.args },
    }));
  }
  return {
    id,
    object: "chat.completion",
    created,
    model: modelId,
    choices: [
      {
        index: 0,
        message,
        finish_reason: openaiStop(answer.finish, calls.length > 0),
      },
    ],
    usage: openaiUsage(answer),
  };
}

/**
 * Streamed OpenAI answer as SSE event blocks (one `data:` line each, the
 * last one `[DONE]`). Role chunk first, content in two halves, tool calls
 * as deltas, then the final chunk carrying finish_reason AND usage.
 */
export function openaiStream(
  answer: UpstreamAnswer,
  modelId: string,
  id: string,
  created: number,
  declared: Set<string>,
): string[] {
  const calls = allowedCalls(answer, declared);
  const stop = openaiStop(answer.finish, calls.length > 0);
  const usage = openaiUsage(answer);
  const chunk = (
    delta: AnyRecord,
    finish: string | null,
    withUsage: boolean,
  ): string => {
    const body: AnyRecord = {
      id,
      object: "chat.completion.chunk",
      created,
      model: modelId,
      choices: [{ index: 0, delta, finish_reason: finish }],
    };
    if (withUsage) body.usage = usage;
    return `data: ${JSON.stringify(body)}`;
  };
  const lines: string[] = [];
  lines.push(chunk({ role: "assistant", content: "" }, null, false));
  const text = answer.text;
  const half = Math.ceil(text.length / 2);
  const first = text.slice(0, half);
  const second = text.slice(half);
  if (first !== "") lines.push(chunk({ content: first }, null, false));
  if (second !== "") lines.push(chunk({ content: second }, null, false));
  calls.forEach((call, i) => {
    lines.push(
      chunk(
        {
          tool_calls: [
            {
              index: i,
              id: toolCallId(call, `${id}-${i}`),
              type: "function",
              function: { name: call.name, arguments: call.args },
            },
          ],
        },
        null,
        false,
      ),
    );
  });
  lines.push(chunk({}, stop, true));
  lines.push("data: [DONE]");
  return lines;
}

/** Non-streamed Anthropic Messages answer. */
export function anthropicResponse(
  answer: UpstreamAnswer,
  modelId: string,
  id: string,
  declared: Set<string>,
): AnyRecord {
  const calls = allowedCalls(answer, declared);
  const content: AnyRecord[] = [];
  if (answer.text !== "") content.push({ type: "text", text: answer.text });
  for (const call of calls) {
    content.push({
      type: "tool_use",
      id: toolCallId(call, `${id}-${content.length}`),
      name: call.name,
      input: parseToolArgs(call.args),
    });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  return {
    id,
    type: "message",
    role: "assistant",
    model: modelId,
    content,
    stop_reason: anthropicStop(answer.finish, calls.length > 0),
    usage: {
      input_tokens: answer.inputTokens,
      output_tokens: answer.outputTokens,
    },
  };
}

/**
 * Streamed Anthropic answer as SSE event blocks (`event:` + `data:` per
 * block). message_start carries the input usage, message_delta the output
 * usage and the stop reason, message_stop closes the stream.
 */
export function anthropicStream(
  answer: UpstreamAnswer,
  modelId: string,
  id: string,
  declared: Set<string>,
): string[] {
  const calls = allowedCalls(answer, declared);
  const stop = anthropicStop(answer.finish, calls.length > 0);
  const event = (name: string, data: AnyRecord): string =>
    `event: ${name}\ndata: ${JSON.stringify(data)}`;
  const blocks: string[] = [];
  blocks.push(
    event("message_start", {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens },
      },
    }),
  );
  let index = 0;
  if (answer.text !== "") {
    blocks.push(
      event("content_block_start", {
        type: "content_block_start",
        index,
        content_block: { type: "text", text: "" },
      }),
    );
    const half = Math.ceil(answer.text.length / 2);
    const first = answer.text.slice(0, half);
    const second = answer.text.slice(half);
    if (first !== "") {
      blocks.push(
        event("content_block_delta", {
          type: "content_block_delta",
          index,
          delta: { type: "text_delta", text: first },
        }),
      );
    }
    if (second !== "") {
      blocks.push(
        event("content_block_delta", {
          type: "content_block_delta",
          index,
          delta: { type: "text_delta", text: second },
        }),
      );
    }
    blocks.push(
      event("content_block_stop", { type: "content_block_stop", index }),
    );
    index += 1;
  }
  for (const call of calls) {
    const callId = toolCallId(call, `${id}-${index}`);
    blocks.push(
      event("content_block_start", {
        type: "content_block_start",
        index,
        content_block: { type: "tool_use", id: callId, name: call.name, input: {} },
      }),
    );
    blocks.push(
      event("content_block_delta", {
        type: "content_block_delta",
        index,
        delta: { type: "input_json_delta", partial_json: call.args },
      }),
    );
    blocks.push(
      event("content_block_stop", { type: "content_block_stop", index }),
    );
    index += 1;
  }
  blocks.push(
    event("message_delta", {
      type: "message_delta",
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  blocks.push(event("message_stop", { type: "message_stop" }));
  return blocks;
}

/** Error body in the wire the client used. */
export function errorBody(
  wire: Wire,
  message: string,
  code: string,
): AnyRecord {
  if (wire === "openai") {
    return {
      error: { message, type: "server_error", code },
    };
  }
  return { type: "error", error: { type: "api_error", message } };
}

/** The failover trail as a header value, or undefined when there is none. */
export function trailHeader(
  trail: { status: number; code: string; model: string | null }[],
): string | undefined {
  if (trail.length === 0) return undefined;
  return JSON.stringify(trail.slice(0, 8));
}
