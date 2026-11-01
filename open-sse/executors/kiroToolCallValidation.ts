import { TEXT_ENCODER } from "./kiro/eventstream.ts";

/**
 * Validation + buffering helpers for Kiro's nested `tool_call` wrapper payloads.
 *
 * Extracted from kiro.ts (file-size gate, #9314) — pure functions, no dependency on
 * KiroExecutor instance state.
 */

export type JsonRecord = Record<string, unknown>;

export const KIRO_TOOL_CALL_WRAPPER = "tool_call";

export type PendingKiroWrapperToolCall = {
  toolCallId: string;
  toolName: string;
  inputKind?: "string" | "object";
  inputText?: string;
  inputObject?: Record<string, unknown>;
};

export function parseKiroToolInput(toolInput: unknown): unknown {
  if (typeof toolInput !== "string") return toolInput;
  try {
    return JSON.parse(toolInput);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid Kiro tool_call payload: input must be valid JSON (${message})`);
  }
}

export function validateKiroToolName(toolUse: JsonRecord): string {
  const toolName = typeof toolUse.name === "string" ? toolUse.name.trim() : "";
  if (!toolName) throw new Error("Invalid Kiro toolUseEvent: missing tool name");
  return toolName;
}

const asSchemaRecord = (value: unknown): JsonRecord | null =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : null;

/**
 * True only when the declared `tool_call` schema is the batched "calls" contract:
 * `calls` is a required array of objects declaring `name` and `arguments`, and the
 * top level does not itself require `name`/`arguments` (legacy contract). A homonymous
 * tool with an optional or differently-shaped `calls` keeps the legacy validation.
 */
export function isKiroCallsContract(schema: unknown): boolean {
  const root = asSchemaRecord(schema);
  const properties = asSchemaRecord(root?.properties);
  const calls = asSchemaRecord(properties?.calls);
  if (!root || calls?.type !== "array") return false;
  const required = Array.isArray(root.required) ? root.required : [];
  if (!required.includes("calls")) return false;
  if (required.includes("name") || required.includes("arguments")) return false;
  const itemProps = asSchemaRecord(asSchemaRecord(calls.items)?.properties);
  return !!itemProps && "name" in itemProps && "arguments" in itemProps;
}

/**
 * True when the wrapper schema declares nothing about its input: the translator
 * synthesizes `{type:"object",properties:{}}` for requests that carry tool_call history
 * but no `tools` (typical of subagent turns). With no declared contract, a batched
 * `{calls:[{name,arguments}]}` payload is as legitimate as the legacy `{name,arguments}`.
 */
export function isKiroUndeclaredWrapperSchema(schema: unknown): boolean {
  const root = asSchemaRecord(schema);
  if (!root) return true;
  const properties = asSchemaRecord(root.properties);
  return !properties || Object.keys(properties).length === 0;
}

/**
 * Decide how to validate a wrapper input. Returns the (possibly normalized) input and
 * whether the batched `calls` contract applies: either the schema declares it, or the
 * schema declares nothing and the input is calls-shaped without a top-level `name`.
 */
export function resolveKiroWrapperInput(
  toolInput: unknown,
  schema: unknown
): { input: unknown; callsContract: boolean } {
  if (isKiroCallsContract(schema)) {
    return { input: normalizeKiroCallsInput(toolInput), callsContract: true };
  }
  if (isKiroUndeclaredWrapperSchema(schema)) {
    const normalized = normalizeKiroCallsInput(toolInput);
    const record = asSchemaRecord(normalized);
    if (record && Array.isArray(record.calls) && !("name" in record)) {
      return { input: normalized, callsContract: true };
    }
  }
  return { input: toolInput, callsContract: false };
}

/**
 * Best-effort normalization of a batched `calls` wrapper input before validation.
 *
 * Models sometimes emit `calls[].arguments` as a JSON string (OpenAI function-call style).
 * A string that parses to a JSON object has an unambiguous meaning, so it is converted to
 * the declared object form instead of failing the whole turn. Anything else (missing/null/
 * empty arguments, missing/blank name, non-JSON string, array/scalar arguments) is returned
 * untouched so the validator still rejects it: the declared contract requires `arguments`.
 */
export function normalizeKiroCallsInput(toolInput: unknown): unknown {
  let input = toolInput;
  if (typeof input === "string") {
    try {
      input = JSON.parse(input);
    } catch {
      return toolInput;
    }
  }
  const record = asSchemaRecord(input);
  if (!record || !Array.isArray(record.calls)) return toolInput;
  const calls = record.calls.map((item: unknown) => {
    const call = asSchemaRecord(item);
    if (!call || typeof call.name !== "string" || !call.name.trim()) return item;
    const args = call.arguments;
    if (typeof args !== "string" || !args.trim()) return item;
    try {
      const parsed = JSON.parse(args);
      return asSchemaRecord(parsed) ? { ...call, arguments: parsed } : item;
    } catch {
      return item;
    }
  });
  return { ...record, calls };
}

export function validateKiroToolCallWrapperInput(toolInput: unknown, callsSchema = false): void {
  if (toolInput === undefined) {
    throw new Error("Invalid Kiro tool_call payload: missing input");
  }
  const input = parseKiroToolInput(toolInput);
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error(
      "Invalid Kiro tool_call payload: input must be an object with name and arguments"
    );
  }
  const record = input as JsonRecord;
  if (callsSchema) {
    if (
      !Array.isArray(record.calls) ||
      record.calls.length === 0 ||
      !record.calls.every((item: unknown) => {
        if (!item || typeof item !== "object" || Array.isArray(item)) return false;
        const call = item as JsonRecord;
        return (
          typeof call.name === "string" &&
          !!call.name.trim() &&
          !!call.arguments &&
          typeof call.arguments === "object" &&
          !Array.isArray(call.arguments)
        );
      })
    ) {
      throw new Error(
        "Invalid Kiro tool_call payload: input.calls must contain named calls with object arguments"
      );
    }
    return;
  }
  if (typeof record.name !== "string" || !record.name.trim()) {
    throw new Error("Invalid Kiro tool_call payload: missing nested MCP tool name at input.name");
  }
  if (!Object.prototype.hasOwnProperty.call(record, "arguments")) {
    throw new Error(
      "Invalid Kiro tool_call payload: missing nested MCP tool arguments at input.arguments"
    );
  }
}

export function validateKiroToolUse(toolUse: JsonRecord): void {
  const toolName = validateKiroToolName(toolUse);
  if (toolName === KIRO_TOOL_CALL_WRAPPER) {
    validateKiroToolCallWrapperInput(toolUse.input);
  }
}

export function appendBufferedKiroToolInput(
  toolCall: PendingKiroWrapperToolCall,
  toolInput: unknown
): void {
  if (toolInput === undefined) return;
  if (typeof toolInput === "string") {
    if (toolCall.inputKind && toolCall.inputKind !== "string") {
      throw new Error("Invalid Kiro tool_call payload: mixed input fragment types");
    }
    toolCall.inputKind = "string";
    toolCall.inputText = `${toolCall.inputText || ""}${toolInput}`;
    return;
  }
  if (toolInput && typeof toolInput === "object" && !Array.isArray(toolInput)) {
    if (toolCall.inputKind && toolCall.inputKind !== "object") {
      throw new Error("Invalid Kiro tool_call payload: mixed input fragment types");
    }
    toolCall.inputKind = "object";
    toolCall.inputObject = toolInput as Record<string, unknown>;
  }
}

export function getBufferedKiroToolInput(toolCall: PendingKiroWrapperToolCall): unknown {
  return toolCall.inputKind === "string" ? toolCall.inputText || "" : toolCall.inputObject;
}

export function encodeSse(value: string): Uint8Array {
  return TEXT_ENCODER.encode(value);
}
