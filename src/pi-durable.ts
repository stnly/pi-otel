/**
 * Pi Durable tracing, exported as `pi-otel/pi-durable`. `src/index.ts` never
 * imports it, so a coding-agent install does not load `@earendil-works/pi-durable`.
 *
 * The host installs the returned extension and passes a Tracer. Generation
 * hooks open `pi.llm_request` (CLIENT). Tool hooks open `pi.tool.<name>` as
 * a sibling and add a link to the llm span that is open for that task.
 * Every span is a root whose trace id is sha256 of the conversation id, so
 * a process that resumes the same conversation continues the trace. Open
 * spans live in memory. A kill drops them. The next request starts a new
 * span on the same trace.
 *
 * `metadata_only` (the default) ships no prompt, completion, or tool payload.
 */

import { createHash, randomBytes } from "node:crypto";
import { ROOT_CONTEXT, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import type { Attributes, Span, SpanContext, Tracer } from "@opentelemetry/api";
import {
  GenerationTask,
  ToolTask,
  defineExtension,
  hook,
} from "@earendil-works/pi-durable";
import type { Extension, ToolExecutionResult } from "@earendil-works/pi-durable";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import {
  ATTR_ERROR_TYPE,
  ATTR_EXCEPTION_MESSAGE,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_CACHE_CREATION_INPUT_TOKENS,
  ATTR_GEN_AI_CACHE_READ_INPUT_TOKENS,
  ATTR_GEN_AI_CACHE_READ_TOKENS,
  ATTR_GEN_AI_CACHE_WRITE_1H_TOKENS,
  ATTR_GEN_AI_CACHE_WRITE_TOKENS,
  ATTR_GEN_AI_COST_USD,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_INPUT_TOKENS,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_TOKENS,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REASONING_OUTPUT_TOKENS,
  ATTR_GEN_AI_REASONING_TOKENS,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_PI_CANCELLED,
  ATTR_PI_TOOL_IS_ERROR,
  GEN_AI_SYSTEM,
  SPAN_LLM_REQUEST,
  clampAttr,
  spanToolName,
  type ContentCapture,
} from "./attrs.js";

/** Not exported by the semconv package this repo pins. */
const ATTR_GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id";

export interface DurableTraceOptions {
  tracer: Tracer;
  /** Default `metadata_only`. `full` ships prompts, completions, and tool payloads. */
  captureContent?: ContentCapture;
  /** `1.43` uses the registry cache and reasoning attribute names. */
  semconv?: "1.36" | "1.37" | "1.43";
}

interface OpenLlm {
  span: Span;
}

interface ConversationTrace {
  llm: Map<string, OpenLlm>;
  /** Span context of the last model call, kept after that span ends so a tool can link to it. */
  lastLlm: SpanContext | undefined;
  tools: Map<string, Span>;
}

/** 32 hex chars. The same conversation id always yields the same trace id. */
export function conversationTraceId(conversationId: string): string {
  return createHash("sha256").update(`pi-otel:${conversationId}`).digest("hex").slice(0, 32);
}

function usageKeys(semconv: "1.36" | "1.37" | "1.43") {
  if (semconv === "1.43") {
    return {
      cacheRead: ATTR_GEN_AI_CACHE_READ_INPUT_TOKENS,
      cacheWrite: ATTR_GEN_AI_CACHE_CREATION_INPUT_TOKENS,
      reasoning: ATTR_GEN_AI_REASONING_OUTPUT_TOKENS,
    };
  }
  return {
    cacheRead: ATTR_GEN_AI_CACHE_READ_TOKENS,
    cacheWrite: ATTR_GEN_AI_CACHE_WRITE_TOKENS,
    reasoning: ATTR_GEN_AI_REASONING_TOKENS,
  };
}

function textOf(message: AssistantMessage): string {
  return message.content
    .flatMap((block) => (block.type === "text" ? [block.text] : []))
    .join("");
}

/** A root span whose trace id is the conversation's, not a freshly minted one. */
function spanContextFor(conversationId: string): SpanContext {
  return {
    traceId: conversationTraceId(conversationId),
    spanId: randomBytes(8).toString("hex"),
    traceFlags: 1,
    isRemote: true,
  };
}

/**
 * Extension the host installs beside its own tools. `endConversation` closes
 * spans a model call left open in this process.
 */
export function createDurableTraces(options: DurableTraceOptions): {
  extension: Extension;
  endConversation(conversationId: string): void;
} {
  const capture = options.captureContent ?? "metadata_only";
  const semconv = options.semconv ?? "1.43";
  const capturePrompt = capture !== "metadata_only";
  const captureTools = capture === "full";
  const keys = usageKeys(semconv);
  const conversations = new Map<string, ConversationTrace>();

  const bucket = (id: string): ConversationTrace => {
    const found = conversations.get(id);
    if (found) return found;
    const created: ConversationTrace = { llm: new Map(), lastLlm: undefined, tools: new Map() };
    conversations.set(id, created);
    return created;
  };

  function start(conversationId: string, name: string, attributes: Attributes, kind: SpanKind): { span: Span } {
    // A remote parent with this trace id makes the SDK continue it instead of minting one.
    const root = trace.setSpanContext(ROOT_CONTEXT, spanContextFor(conversationId));
    const span = options.tracer.startSpan(
      name,
      {
        kind,
        attributes: {
          [ATTR_GEN_AI_AGENT_NAME]: GEN_AI_SYSTEM,
          [ATTR_GEN_AI_CONVERSATION_ID]: conversationId,
          ...attributes,
        },
      },
      root,
    );
    return { span };
  }

  const extension = defineExtension({
    name: "pi-otel",
    hooks: [
      hook(GenerationTask, {
        beforeRequest(request, api) {
          const conversationId = String(api.conversationId);
          const taskId = String(api.taskId);
          const state = bucket(conversationId);
          state.llm.get(taskId)?.span.end();
          const attributes: Attributes = { [ATTR_GEN_AI_OPERATION_NAME]: "chat" };
          if (capturePrompt && request.messages.length > 0) {
            attributes[ATTR_GEN_AI_INPUT_MESSAGES] = clampAttr(request.messages);
          }
          const opened = start(conversationId, SPAN_LLM_REQUEST, attributes, SpanKind.CLIENT);
          opened.span.setAttribute("pi.durable.task_id", taskId);
          state.llm.set(taskId, opened);
          return undefined;
        },
        afterResponse(message, api) {
          const state = conversations.get(String(api.conversationId));
          const opened = state?.llm.get(String(api.taskId));
          if (!state || !opened) return;
          state.lastLlm = opened.span.spanContext();
          state.llm.delete(String(api.taskId));
          finishLlm(opened.span, message, capturePrompt, keys);
        },
      }),
      hook(ToolTask, {
        beforeTool(call, api) {
          const conversationId = String(api.conversationId);
          const state = bucket(conversationId);
          const attributes: Attributes = {
            [ATTR_GEN_AI_OPERATION_NAME]: "execute_tool",
            [ATTR_GEN_AI_TOOL_NAME]: call.name,
            [ATTR_GEN_AI_TOOL_CALL_ID]: call.id,
          };
          if (captureTools) attributes[ATTR_GEN_AI_TOOL_CALL_ARGUMENTS] = clampAttr(call.arguments);
          const opened = start(conversationId, spanToolName(call.name), attributes, SpanKind.INTERNAL);
          if (state.lastLlm?.traceId && state.lastLlm.spanId) opened.span.addLink({ context: state.lastLlm });
          state.tools.set(call.id, opened.span);
          return undefined;
        },
        afterTool(call, result, api) {
          const state = conversations.get(String(api.conversationId));
          const span = state?.tools.get(call.id);
          if (!state || !span) return undefined;
          state.tools.delete(call.id);
          finishTool(span, result, captureTools);
          return undefined;
        },
      }),
    ],
  });

  return {
    extension,
    endConversation(conversationId) {
      const state = conversations.get(conversationId);
      if (!state) return;
      conversations.delete(conversationId);
      for (const opened of state.llm.values()) {
        opened.span.setAttribute(ATTR_PI_CANCELLED, true);
        opened.span.end();
      }
      for (const span of state.tools.values()) {
        span.setAttribute(ATTR_PI_CANCELLED, true);
        span.end();
      }
    },
  };
}

function finishLlm(
  span: Span,
  message: AssistantMessage,
  capturePrompt: boolean,
  keys: { cacheRead: string; cacheWrite: string; reasoning: string },
): void {
  span.setAttribute(ATTR_GEN_AI_PROVIDER_NAME, message.provider);
  span.setAttribute(ATTR_GEN_AI_REQUEST_MODEL, message.model);
  if (message.responseModel) span.setAttribute(ATTR_GEN_AI_RESPONSE_MODEL, message.responseModel);
  if (message.responseId) span.setAttribute(ATTR_GEN_AI_RESPONSE_ID, message.responseId);
  span.setAttribute(ATTR_GEN_AI_RESPONSE_FINISH_REASONS, [message.stopReason]);
  const usage = message.usage;
  if (usage) {
    span.setAttribute(ATTR_GEN_AI_INPUT_TOKENS, usage.input);
    span.setAttribute(ATTR_GEN_AI_OUTPUT_TOKENS, usage.output);
    span.setAttribute(keys.cacheRead, usage.cacheRead);
    span.setAttribute(keys.cacheWrite, usage.cacheWrite);
    if (typeof usage.reasoning === "number") span.setAttribute(keys.reasoning, usage.reasoning);
    if (typeof usage.cacheWrite1h === "number") {
      span.setAttribute(ATTR_GEN_AI_CACHE_WRITE_1H_TOKENS, usage.cacheWrite1h);
    }
    if (typeof usage.cost?.total === "number") span.setAttribute(ATTR_GEN_AI_COST_USD, usage.cost.total);
  }
  if (capturePrompt) {
    const text = textOf(message);
    span.setAttribute(
      ATTR_GEN_AI_OUTPUT_MESSAGES,
      clampAttr([{ role: "assistant", parts: text ? [{ type: "text", content: text }] : [], finish_reason: message.stopReason }]),
    );
  }
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    span.setAttribute(ATTR_ERROR_TYPE, message.stopReason);
    span.setAttribute(ATTR_PI_CANCELLED, message.stopReason === "aborted");
    if (message.errorMessage) span.setAttribute(ATTR_EXCEPTION_MESSAGE, clampAttr(message.errorMessage));
    span.setStatus({ code: SpanStatusCode.ERROR, message: message.stopReason });
  }
  span.end();
}

function finishTool(span: Span, result: ToolExecutionResult, captureTools: boolean): void {
  const isError = result.isError === true;
  span.setAttribute(ATTR_PI_TOOL_IS_ERROR, isError);
  if (isError) {
    span.setAttribute(ATTR_ERROR_TYPE, "tool_error");
    span.setStatus({ code: SpanStatusCode.ERROR, message: "tool_error" });
  }
  if (captureTools && result.output !== undefined) {
    span.setAttribute(ATTR_GEN_AI_TOOL_CALL_RESULT, clampAttr(result.output));
  }
  span.end();
}
