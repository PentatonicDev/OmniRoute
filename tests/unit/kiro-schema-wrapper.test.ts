import test from "node:test";
import assert from "node:assert/strict";

import { KiroExecutor } from "../../open-sse/executors/kiro.ts";
import { isKiroCallsContract } from "../../open-sse/executors/kiroToolCallValidation.ts";
import { FORMATS } from "../../open-sse/translator/formats.ts";
import { createSSETransformStreamWithLogger } from "../../open-sse/utils/stream.ts";

const textEncoder = new TextEncoder();

function crc32(bytes: Uint8Array): number {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let value = i;
    for (let bit = 0; bit < 8; bit++) {
      value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[i] = value >>> 0;
  }

  let value = 0xffffffff;
  for (const byte of bytes) {
    value = table[(value ^ byte) & 0xff] ^ (value >>> 8);
  }
  return (value ^ 0xffffffff) >>> 0;
}

function encodeHeader(name: string, value: string): Uint8Array {
  const nameBytes = textEncoder.encode(name);
  const valueBytes = textEncoder.encode(value);
  const header = new Uint8Array(1 + nameBytes.length + 1 + 2 + valueBytes.length);
  let offset = 0;
  header[offset++] = nameBytes.length;
  header.set(nameBytes, offset);
  offset += nameBytes.length;
  header[offset++] = 7;
  header[offset++] = (valueBytes.length >> 8) & 0xff;
  header[offset++] = valueBytes.length & 0xff;
  header.set(valueBytes, offset);
  return header;
}

function encodeEventFrame(eventType: string, payload: Record<string, unknown>): Uint8Array {
  const headers = encodeHeader(":event-type", eventType);
  const payloadBytes = textEncoder.encode(JSON.stringify(payload));
  const totalLength = 12 + headers.length + payloadBytes.length + 4;
  const frame = new Uint8Array(totalLength);
  const view = new DataView(frame.buffer);
  view.setUint32(0, totalLength, false);
  view.setUint32(4, headers.length, false);
  view.setUint32(8, crc32(frame.slice(0, 8)), false);
  frame.set(headers, 12);
  frame.set(payloadBytes, 12 + headers.length);
  view.setUint32(totalLength - 4, crc32(frame.slice(0, totalLength - 4)), false);
  return frame;
}

function buildEventStreamResponse(frames: Uint8Array[]): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const frame of frames) controller.enqueue(frame);
        controller.close();
      },
    }),
    { status: 200 }
  );
}

function collectSseJson(text: string): Array<Record<string, unknown>> {
  return text
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => line.slice(6).trim())
    .filter((line) => line && line !== "[DONE]")
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function toolDeltas(text: string): Array<Record<string, unknown>> {
  return collectSseJson(text).flatMap((chunk) => {
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = choices[0] as Record<string, unknown> | undefined;
    const delta = choice?.delta as Record<string, unknown> | undefined;
    return Array.isArray(delta?.tool_calls)
      ? (delta.tool_calls as Array<Record<string, unknown>>)
      : [];
  });
}

test("declared calls wrapper emits valid synthetic Hermes calls at Kiro seam", async () => {
  const executor = new KiroExecutor();
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", {
      toolUseId: "synthetic_1",
      name: "tool_call",
      input: { calls: [{ name: "synthetic_lookup", arguments: {} }] },
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);
  const schema = {
    type: "object",
    properties: {
      calls: {
        type: "array",
        items: {
          type: "object",
          properties: { name: { type: "string" }, arguments: { type: "object" } },
          required: ["name", "arguments"],
        },
      },
    },
    required: ["calls"],
  };
  const text = await executor
    .transformEventStreamToSSE(response, "synthetic-model", { toolCallSchema: schema })
    .text();
  assert.doesNotMatch(text, /invalid_kiro_tool_call/);
  assert.equal(toolDeltas(text).length, 2);
});

test("DIAGNOSTIC synthetic MCP name and arguments is accepted at Kiro seam", async () => {
  const executor = new KiroExecutor();
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", {
      toolUseId: "synthetic_2",
      name: "tool_call",
      input: { name: "synthetic_lookup", arguments: {} },
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);
  const text = await executor.transformEventStreamToSSE(response, "synthetic-model").text();
  assert.doesNotMatch(text, /invalid_kiro_tool_call/);
  assert.equal(toolDeltas(text).length, 2);
});

test("declared calls validates each call and preserves multiple fragmented calls", async () => {
  const schema = HERMES_SCHEMA;
  for (const input of [
    { calls: [] },
    { calls: [{ name: "", arguments: {} }] },
    { calls: [{ name: "ok", arguments: null }] },
    { calls: [{ name: "ok" }] },
  ]) {
    const response = buildEventStreamResponse([
      encodeEventFrame("toolUseEvent", { toolUseId: "1", name: "tool_call", input }),
      encodeEventFrame("messageStopEvent", {}),
    ]);
    const text = await new KiroExecutor()
      .transformEventStreamToSSE(response, "model", { toolCallSchema: schema })
      .text();
    assert.match(text, /invalid_kiro_tool_call/);
  }
  const expected = {
    calls: [
      { name: "first", arguments: {} },
      { name: "second", arguments: { value: 1 } },
    ],
  };
  const serialized = JSON.stringify(expected);
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", {
      toolUseId: "1",
      name: "tool_call",
      input: serialized.slice(0, 20),
    }),
    encodeEventFrame("toolUseEvent", {
      toolUseId: "1",
      name: "tool_call",
      input: serialized.slice(20),
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);
  const text = await new KiroExecutor()
    .transformEventStreamToSSE(response, "model", { toolCallSchema: schema })
    .text();
  const args = toolDeltas(text)
    .map((x) => (x.function as Record<string, unknown>).arguments || "")
    .join("");
  assert.deepEqual(JSON.parse(args), expected);
});

test("homonymous tool without calls schema retains original validation", async () => {
  const response = buildEventStreamResponse([
    encodeEventFrame("toolUseEvent", {
      toolUseId: "1",
      name: "tool_call",
      input: { calls: [{ name: "one", arguments: {} }] },
    }),
    encodeEventFrame("messageStopEvent", {}),
  ]);
  const text = await new KiroExecutor()
    .transformEventStreamToSSE(response, "model", {
      toolCallSchema: { properties: { other: { type: "string" } } },
    })
    .text();
  assert.match(text, /invalid_kiro_tool_call/);
});

const HERMES_SCHEMA = {
  type: "object",
  properties: {
    calls: {
      type: "array",
      items: {
        type: "object",
        properties: { name: { type: "string" }, arguments: { type: "object" } },
        required: ["name", "arguments"],
      },
    },
  },
  required: ["calls"],
};

test("legacy tool_call with optional unrelated calls property keeps name/arguments contract", async () => {
  const legacy = {
    type: "object",
    properties: {
      name: { type: "string" },
      arguments: { type: "object" },
      calls: { type: "array", items: { type: "string" } },
    },
    required: ["name", "arguments"],
  };
  const frames = (input: unknown) => [
    encodeEventFrame("toolUseEvent", { toolUseId: "1", name: "tool_call", input }),
    encodeEventFrame("messageStopEvent", {}),
  ];
  const valid = await new KiroExecutor()
    .transformEventStreamToSSE(
      buildEventStreamResponse(frames({ name: "synthetic_lookup", arguments: {} })),
      "model",
      { toolCallSchema: legacy }
    )
    .text();
  assert.doesNotMatch(valid, /invalid_kiro_tool_call/);
  assert.equal(toolDeltas(valid).length, 2);

  const wrongContract = await new KiroExecutor()
    .transformEventStreamToSSE(
      buildEventStreamResponse(frames({ calls: [{ name: "x", arguments: {} }] })),
      "model",
      { toolCallSchema: legacy }
    )
    .text();
  assert.match(wrongContract, /invalid_kiro_tool_call/);
});

test("isKiroCallsContract recognizes only the required calls[] contract", () => {
  assert.equal(isKiroCallsContract(HERMES_SCHEMA), true);
  assert.equal(isKiroCallsContract(undefined), false);
  assert.equal(isKiroCallsContract({ properties: { calls: { type: "array" } } }), false);
  assert.equal(
    isKiroCallsContract({ ...HERMES_SCHEMA, required: ["calls", "name", "arguments"] }),
    false
  );
  assert.equal(
    isKiroCallsContract({
      type: "object",
      properties: { calls: { type: "array", items: { type: "string" } } },
      required: ["calls"],
    }),
    false
  );
});

async function translateToClient(executorSse: string, clientFormat: string): Promise<string> {
  const transform = createSSETransformStreamWithLogger(
    FORMATS.OPENAI,
    clientFormat,
    "kiro",
    null,
    null,
    "synthetic-model"
  );
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(textEncoder.encode(executorSse));
      controller.close();
    },
  });
  const reader = upstream.pipeThrough(transform).getReader();
  let out = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out += new TextDecoder().decode(value);
  }
  return out;
}

const CALLS_CASES: Array<[string, unknown, string[]]> = [
  ["simple", { calls: [{ name: "synthetic_lookup", arguments: {} }] }, []],
  [
    "multiple",
    {
      calls: [
        { name: "first", arguments: { a: 1 } },
        { name: "second", arguments: { b: [2] } },
      ],
    },
    [],
  ],
  [
    "fragmented",
    {
      calls: [
        { name: "first", arguments: {} },
        { name: "second", arguments: { v: 1 } },
      ],
    },
    ["fragment"],
  ],
];

for (const [label, expected, flags] of CALLS_CASES) {
  for (const client of [FORMATS.OPENAI, FORMATS.CLAUDE]) {
    test(`calls[] ${label} survives executor + ${client} client translation`, async () => {
      const serialized = JSON.stringify(expected);
      const inputs = flags.includes("fragment")
        ? [serialized.slice(0, 17), serialized.slice(17, 40), serialized.slice(40)]
        : [expected];
      const executorSse = await new KiroExecutor()
        .transformEventStreamToSSE(
          buildEventStreamResponse([
            ...inputs.map((input) =>
              encodeEventFrame("toolUseEvent", { toolUseId: "t1", name: "tool_call", input })
            ),
            encodeEventFrame("messageStopEvent", {}),
          ]),
          "synthetic-model",
          { toolCallSchema: HERMES_SCHEMA }
        )
        .text();
      assert.doesNotMatch(executorSse, /invalid_kiro_tool_call/);
      const text = await translateToClient(executorSse, client);
      assert.doesNotMatch(text, /invalid_kiro_tool_call/);
      let args = "";
      if (client === FORMATS.OPENAI) {
        args = toolDeltas(text)
          .map((x) => (x.function as Record<string, unknown>).arguments || "")
          .join("");
        assert.ok(
          toolDeltas(text).some(
            (x) => (x.function as Record<string, unknown> | undefined)?.name === "tool_call"
          )
        );
      } else {
        args = collectSseJson(text)
          .filter(
            (c) => (c.delta as Record<string, unknown> | undefined)?.type === "input_json_delta"
          )
          .map((c) => (c.delta as Record<string, unknown>).partial_json as string)
          .join("");
        assert.match(text, /"name":"tool_call"/);
      }
      assert.deepEqual(JSON.parse(args), expected);
    });
  }
}
