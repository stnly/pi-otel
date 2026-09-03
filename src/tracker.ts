/**
 * Span lifecycle tracker.
 *
 * Owns the span tree:
 *   pi.interaction (per user prompt)
 *   └─ pi.turn (per LLM call + tool execution)
 *      ├─ pi.llm_request  [SpanKind.CLIENT]  (the model call)
 *      └─ pi.tool.<name>                     (siblings: each tool call)
 *
 * `pi.session` is the root, opened on session_start and closed on
 * session_shutdown. Within it, each user prompt opens an interaction.
 *
 * Design notes:
 *  - Tool spans are siblings of the LLM span under the turn, NOT children.
 *    Tools execute after the model call returns, so parenting them to the
 *    LLM span would misrepresent the causal relationship.
 *  - Every span is closed defensively: on turn_end, on session replacement
 *    (before_switch/fork/compact/tree), on session_shutdown, and on abort.
 *    No orphaned spans survive.
 *  - Cancellation (Esc/abort) is detected via ctx.signal during a turn and
 *    marks active spans with `pi.cancelled` and ERROR status.
 *  - All handlers are best-effort: telemetry must never break pi's agent loop.
 */

import {
  type Context,
  SpanKind,
  SpanStatusCode,
  type Span,
  type SpanContext,
  type Tracer,
  context as otelContext,
  trace,
  type Attributes,
} from "@opentelemetry/api";
// Message shapes are imported structurally (see MessageShapes below) rather
// than from @earendil-works/pi-ai, which is a transitive dep of pi-coding-agent
// and not guaranteed to resolve from an extension's own node_modules.
import {
  ATTR_GEN_AI_AGENT_NAME,
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
  ATTR_GEN_AI_REASONING_TOKENS,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_SYSTEM_PROMPT_HASH,
  ATTR_GEN_AI_TOKEN_TYPE,
  GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_CALL_RESULT,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_HTTP_STATUS_CODE,
  ATTR_ERROR_TYPE,
  ATTR_EXCEPTION_MESSAGE,
  ATTR_PI_CANCELLED,
  ATTR_PI_CWD,
  ATTR_PI_ERROR_COUNT,
  ATTR_PI_INTERACTION_ID,
  ATTR_PI_INTERACTION_ORIGIN,
  ATTR_PI_ORPHANED,
  ATTR_PI_PROMPT_LENGTH,
  ATTR_PI_SESSION_FILE,
  ATTR_PI_SESSION_ID,
  ATTR_PI_SESSION_PARENT_ID,
  ATTR_PI_SESSION_REASON,
  ATTR_PI_TOOL_COUNT,
  ATTR_PI_TOOL_IS_ERROR,
  ATTR_PI_TURN_COUNT,
  ATTR_PI_TURN_INDEX,
  ATTR_PI_USER_PROMPT,
  clampAttr,
  EVENT_GEN_AI_ASSISTANT_MESSAGE,
  EVENT_GEN_AI_CHOICE,
  EVENT_GEN_AI_COMPLETION,
  EVENT_GEN_AI_FIRST_TOKEN,
  EVENT_GEN_AI_TOOL_MESSAGE,
  EVENT_GEN_AI_USER_MESSAGE,
  hashPrompt,
  SPAN_INTERACTION,
  SPAN_LLM_REQUEST,
  SPAN_SESSION,
  SPAN_TURN,
  fingerprint,
  spanToolName,
  type ContentCapture,
} from "./attrs.js";
import type { Metrics } from "./metrics.js";
import type { SemconvDialect } from "./config.js";

export interface TrackerOptions {
  tracer: Tracer;
  captureContent: ContentCapture;
  /** GenAI convention version. Default `1.37` (gen_ai.provider.name, no
   * message span events). `1.36` emits the pre-2025-10 set (gen_ai.system
   * plus the message events) for backends not yet migrated. */
  semconv?: SemconvDialect;
  /** Lazy session id so the tracker doesn't need ctx at construction. */
  sessionId: () => string | undefined;
  sessionFile: () => string | undefined;
  cwd: string;
  metrics: () => Metrics | null;
  /** Orphan sweep interval in ms. Default 60_000. */
  orphanSweepIntervalMs?: number;
  /** Age at which an open span is considered orphaned, in ms. Default 30 * 60 * 1000. */
  orphanTtlMs?: number;
  /** Injection point for tests. Defaults to setInterval / clearInterval / Date.now. */
  now?: () => number;
  setTimer?: (fn: () => void, ms: number) => () => void;
}

interface Slot {
  span: Span;
  ctx: Context;
}
interface TimedSlot extends Slot {
  /** Monotonic nanoseconds (process.hrtime.bigint) for accurate sub-ms durations. */
  startNs: bigint;
  /** Wall-clock ms (injectable now()) for orphan-sweep age checks. */
  startMs: number;
}
interface ToolSlot extends TimedSlot {
  name: string;
}

const OP_NAME_CHAT = "chat";

function categorizeHttpError(status: number): string {
  if (status === 408) return "timeout";
  if (status === 413) return "request_too_large";
  if (status === 429) return "rate_limit";
  if (status === 401 || status === 403) return "auth_error";
  if (status >= 500) return "server_error";
  // Only reached for unexpected 4xx statuses this extension does not name.
  return "client_error";
}

function categorizeThrownError(combined: string, errName: string): string {
  if (/timeout|abort/i.test(combined)) return "timeout";
  if (/rate.?limit/i.test(combined)) return "rate_limit";
  if (/auth|unauthorized|forbidden/i.test(combined)) return "auth_error";
  if (/context.*(length|window)|too.*(large|long)/i.test(combined)) return "request_too_large";
  if (/content.?filter|safety/i.test(combined)) return "content_filter";
  return errName;
}

export type SessionReason = "startup" | "reload" | "new" | "resume" | "fork";

export class SpanTracker {
  private opts: TrackerOptions;
  private readonly semconv: SemconvDialect;
  private session: (Slot) | null = null;
  private interaction: (Slot) | null = null;
  private turn: (TimedSlot & { index: number }) | null = null;
  private llm: (TimedSlot & {
    requestModel?: string;
    providerSystem?: string;
    responseModel?: string;
    toolCallCount?: number;
    inputTokens?: number;
    outputTokens?: number;
    reasoningTokens?: number;
    cacheReadTokens?: number;
    cacheWriteTokens?: number;
    cacheWrite1hTokens?: number;
    httpStatus?: number;
    attempts: number;
    /** Failed HTTP attempts seen so far (reset never; recovery clears lastError*). */
    failedAttempts: number;
    /** Category of the most recent failed attempt; cleared by a successful one. */
    lastErrorType?: string;
    /** HTTP status of the most recent failed attempt. */
    lastErrorStatus?: number;
    inputMessages: Array<Record<string, unknown>>;
    firstTokenSeen?: boolean;
    completionRecorded?: boolean;
  }) | null = null;
  private tools = new Map<string, ToolSlot>();

  private interactionCount = 0;
  /** Turns in the current interaction (reset each startInteraction). */
  private interactionTurnCount = 0;
  /** Tools in the current interaction (reset each startInteraction). */
  private interactionToolCount = 0;
  /** Session-lifetime turn total (never reset until endSession). */
  private sessionTurnCount = 0;
  /** Session-lifetime tool total (never reset until endSession). */
  private sessionToolCount = 0;
  private sessionStartMs = 0;
  private orphanTimer: (() => void) | null = null;
  private readonly now: () => number;
  private readonly orphanTtlMs: number;
  private interactionStartMs = 0;
  private totalInputTokens = 0;
  private totalOutputTokens = 0;
  private totalCostUsd = 0;
  private errorCount = 0;
  /** Last error reference counted toward errorCount, for cascade dedupe. */
  private lastCountedError: unknown = undefined;

  constructor(opts: TrackerOptions) {
    this.opts = opts;
    this.semconv = opts.semconv ?? "1.37";
    this.now = opts.now ?? (() => Date.now());
    this.orphanTtlMs = opts.orphanTtlMs ?? 30 * 60 * 1000;
    this.setTimer = opts.setTimer ?? defaultSetTimer;
  }

  private setTimer: (fn: () => void, ms: number) => () => void;

  // ---------------------------------------------------------------- sessions
  startSession(reason?: SessionReason, parentId?: string): void {
    if (this.session) this.endSession();
    this.sessionStartMs = this.now();
    const attrs = this.commonAttrs();
    if (reason !== undefined) {
      attrs[ATTR_PI_SESSION_REASON] = reason;
    }
    if (parentId) {
      attrs[ATTR_PI_SESSION_PARENT_ID] = parentId;
    }
    const span = this.opts.tracer.startSpan(SPAN_SESSION, {
      attributes: attrs,
    });
    this.session = {
      span,
      ctx: trace.setSpan(otelContext.active(), span),
    };
    this.startOrphanSweep();
  }

  endSession(): void {
    // Defensive: close everything still open.
    this.endInteraction({ reason: "session_end" });
    if (this.sessionStartMs > 0) {
      const durSec = (this.now() - this.sessionStartMs) / 1000;
      try {
        this.opts.metrics()?.sessionDuration.record(durSec, this.commonAttrs());
      } catch { /* best-effort */ }
    }
    if (this.session) {
      this.session.span.setAttribute(ATTR_PI_TURN_COUNT, this.sessionTurnCount);
      this.session.span.setAttribute(ATTR_PI_TOOL_COUNT, this.sessionToolCount);
      if (this.totalInputTokens > 0) {
        this.session.span.setAttribute(ATTR_GEN_AI_INPUT_TOKENS, this.totalInputTokens);
      }
      if (this.totalOutputTokens > 0) {
        this.session.span.setAttribute(ATTR_GEN_AI_OUTPUT_TOKENS, this.totalOutputTokens);
      }
      if (this.totalCostUsd > 0) {
        this.session.span.setAttribute(ATTR_GEN_AI_COST_USD, this.totalCostUsd);
      }
      if (this.errorCount > 0) {
        this.session.span.setAttribute(ATTR_PI_ERROR_COUNT, this.errorCount);
      }
      this.session.span.end();
      this.session = null;
    }
    this.sessionTurnCount = 0;
    this.sessionToolCount = 0;
    this.interactionTurnCount = 0;
    this.interactionToolCount = 0;
    this.pendingInput = [];
    this.stopOrphanSweep();
  }

  /**
   * Periodically end any span open longer than orphanTtlMs. Belt-and-suspenders
   * over the defensive closes in endInteraction: catches spans orphaned by
   * unexpected code paths or a process that skips session_shutdown.
   */
  private startOrphanSweep(): void {
    if (this.orphanTimer) return;
    const intervalMs = this.opts.orphanSweepIntervalMs ?? 60_000;
    this.orphanTimer = this.setTimer(() => this.sweepOrphans(), intervalMs);
  }

  private stopOrphanSweep(): void {
    if (this.orphanTimer) {
      this.orphanTimer();
      this.orphanTimer = null;
    }
  }

  private sweepOrphans(): void {
    const cutoff = this.now() - this.orphanTtlMs;
    // End any tool span older than the TTL.
    for (const [id, slot] of this.tools) {
      const startMs = slot.startMs;
      if (startMs < cutoff) {
        slot.span.setAttribute(ATTR_PI_ORPHANED, true);
        slot.span.end();
        this.tools.delete(id);
      }
    }
    // LLM span.
    if (this.llm && this.llm.startMs < cutoff) {
      this.llm.span.setAttribute(ATTR_PI_ORPHANED, true);
      this.applyLlmErrorState(this.llm.span);
      this.llm.span.end();
      this.llm = null;
    }
    // Turn span.
    if (this.turn && this.turn.startMs < cutoff) {
      this.turn.span.setAttribute(ATTR_PI_ORPHANED, true);
      this.turn.span.end();
      this.turn = null;
    }
    // Interaction span.
    if (this.interaction && this.interactionStartMs < cutoff) {
      this.interaction.span.setAttribute(ATTR_PI_ORPHANED, true);
      this.interaction.span.end();
      this.interaction = null;
      this.interactionStartMs = 0;
    }
  }

  // ----------------------------------------------------------- interactions
  /** A user prompt begins an interaction. */
  startInteraction(prompt: string | undefined): void {
    if (this.interaction) this.endInteraction({ reason: "superseded" });
    this.openInteraction(prompt, "user");
  }

  /**
   * Open an interaction only when none is open. Covers runs pi starts
   * without `before_agent_start`: the retry after an `agent_end` that closed
   * the interaction (pi < 0.80.5), and extension-triggered turns on an idle
   * session. Not a user prompt: no prompt metric, no prompt attributes; the
   * span is tagged `pi.interaction.origin=agent`.
   */
  ensureInteraction(): void {
    if (this.interaction) return;
    this.openInteraction(undefined, "agent");
  }

  private openInteraction(prompt: string | undefined, origin: "user" | "agent"): void {
    this.interactionCount++;
    this.interactionTurnCount = 0;
    this.interactionToolCount = 0;
    this.pendingInput = [];
    const attrs = this.commonAttrs();
    attrs[ATTR_PI_INTERACTION_ID] = this.interactionCount;
    attrs[ATTR_PI_INTERACTION_ORIGIN] = origin;
    if (typeof prompt === "string") {
      attrs[ATTR_PI_PROMPT_LENGTH] = prompt.length;
      if (this.shouldCapturePrompt()) {
        attrs[ATTR_PI_USER_PROMPT] = clampAttr(prompt);
      } else {
        Object.assign(attrs, prefixKeys("pi.user_prompt", fingerprint(prompt)));
      }
    }
    const parent = this.session?.ctx ?? otelContext.active();
    const span = this.opts.tracer.startSpan(SPAN_INTERACTION, { attributes: attrs }, parent);
    this.interaction = { span, ctx: trace.setSpan(parent, span) };
    this.interactionStartMs = this.now();
    if (origin === "user") {
      try { this.opts.metrics()?.promptCount.add(1, this.commonAttrs()); } catch { /* noop */ }
    }
  }

  endInteraction(opts: { reason: string; cancelled?: boolean; error?: unknown } = { reason: "end" }): void {
    // Close any in-flight LLM/tool/turn first.
    this.endLlm({ reason: opts.reason, cancelled: opts.cancelled, error: opts.error });
    this.endAllTools(opts.reason);
    this.endTurn({ reason: opts.reason, cancelled: opts.cancelled, error: opts.error });
    if (this.interaction) {
      const span = this.interaction.span;
      if (opts.cancelled) {
        span.setAttribute(ATTR_PI_CANCELLED, true);
      } else if (opts.reason !== "end") {
        // Any non-normal end (superseded, session_switch/fork/tree, session_end)
        // means the interaction was abandoned mid-flight.
        span.setAttribute(ATTR_PI_ORPHANED, true);
      }
      span.setAttribute(ATTR_PI_TURN_COUNT, this.interactionTurnCount);
      span.setAttribute(ATTR_PI_TOOL_COUNT, this.interactionToolCount);
      this.setStatusFromError(span, opts.error);
      span.end();
      this.interaction = null;
      this.interactionStartMs = 0;
      this.pendingInput = [];
    }
  }

  // ------------------------------------------------------------------ turns
  startTurn(turnIndex: number): void {
    if (!this.interaction) return;
    if (this.turn) this.endTurn({ reason: "superseded" });
    this.interactionTurnCount++;
    this.sessionTurnCount++;
    const attrs = this.commonAttrs();
    attrs[ATTR_PI_TURN_INDEX] = turnIndex;
    attrs[ATTR_GEN_AI_OPERATION_NAME] = OP_NAME_CHAT;
    const span = this.opts.tracer.startSpan(SPAN_TURN, { attributes: attrs }, this.interaction.ctx);
    this.turn = { span, ctx: trace.setSpan(this.interaction.ctx, span), index: turnIndex, startNs: process.hrtime.bigint(), startMs: this.now() };
    try { this.opts.metrics()?.turnCount.add(1, this.commonAttrs()); } catch { /* noop */ }
  }

  endTurn(opts: { reason: string; cancelled?: boolean; error?: unknown } = { reason: "end" }): void {
    if (!this.turn) return;
    const span = this.turn.span;
    if (opts.cancelled) span.setAttribute(ATTR_PI_CANCELLED, true);
    else if (opts.reason === "superseded") span.setAttribute(ATTR_PI_ORPHANED, true);
    this.setStatusFromError(span, opts.error);
    span.end();
    this.turn = null;
    // The turn-cancellation metric is bumped from markCancelled so it only
    // counts aborts that actually cancelled an in-flight turn. The cancelled
    // flag here only drives the span attribute.
  }

  // ------------------------------------------------------------- llm spans
  startLlm(requestModel: string | undefined, providerSystem: string | undefined): void {
    if (this.llm) this.endLlm({ reason: "superseded" });
    const parent = this.turn?.ctx ?? this.interaction?.ctx ?? otelContext.active();
    const attrs = this.commonAttrs();
    attrs[ATTR_GEN_AI_OPERATION_NAME] = OP_NAME_CHAT;
    // Agent identity is the harness (pi). Provider identity is the dialect's
    // provider key: gen_ai.provider.name in 1.37 (the 2025-10 rename,
    // semantic-conventions v1.37.0), gen_ai.system in 1.36. Each dialect
    // emits only its own key; a dialect that also wrote the other version's
    // key would not be that version.
    attrs[ATTR_GEN_AI_AGENT_NAME] = GEN_AI_SYSTEM;
    if (providerSystem) {
      attrs[this.providerNameKey()] = providerSystem;
    }
    if (requestModel) attrs[ATTR_GEN_AI_REQUEST_MODEL] = requestModel;
    const span = this.opts.tracer.startSpan(
      SPAN_LLM_REQUEST,
      { kind: SpanKind.CLIENT, attributes: attrs },
      parent,
    );
    this.llm = {
      span,
      ctx: trace.setSpan(parent, span),
      startNs: process.hrtime.bigint(),
      startMs: this.now(),
      requestModel,
      providerSystem,
      attempts: 0,
      failedAttempts: 0,
      inputMessages: [],
    };
    // Drain any user/tool messages that arrived before the LLM span opened.
    this.flushPendingInput();
  }

  noteFirstToken(message: { role?: string }): void {
    if (!this.llm || message.role !== "assistant" || this.llm.firstTokenSeen) return;
    this.llm.firstTokenSeen = true;
    const elapsedSec = Number(process.hrtime.bigint() - this.llm.startNs) / 1e9;
    const base: Attributes = this.commonAttrs();
    if (this.llm.requestModel) base[ATTR_GEN_AI_REQUEST_MODEL] = this.llm.requestModel;
    try {
      this.opts.metrics()?.timeToFirstToken.record(elapsedSec, base);
    } catch { /* best-effort */ }
    this.llm.span.addEvent(EVENT_GEN_AI_FIRST_TOKEN, { elapsed_s: elapsedSec } as Attributes);
  }

  noteLlmComplete(message: { role?: string }): void {
    if (!this.llm || message.role !== "assistant" || this.llm.completionRecorded) return;
    this.llm.completionRecorded = true;
    const elapsedSec = Number(process.hrtime.bigint() - this.llm.startNs) / 1e9;
    const base: Attributes = this.commonAttrs();
    if (this.llm.requestModel) base[ATTR_GEN_AI_REQUEST_MODEL] = this.llm.requestModel;
    try {
      this.opts.metrics()?.timeToCompletion.record(elapsedSec, base);
    } catch { /* best-effort */ }
    this.llm.span.addEvent(EVENT_GEN_AI_COMPLETION, { elapsed_s: elapsedSec } as Attributes);
  }

  noteSystemPrompt(prompt: string): void {
    const hash = hashPrompt(prompt);
    if (!hash) return;
    const target = this.interaction?.span ?? this.session?.span;
    if (target) target.setAttribute(ATTR_GEN_AI_SYSTEM_PROMPT_HASH, hash);
  }

  /** Buffer an input message (user or tool result) to flush when the LLM span opens. */
  private pendingInput: Array<{ role: "user" | "tool"; text: string; toolCallId?: string; toolName?: string }> = [];

  noteUserInput(text: string): void {
    this.pendingInput.push({ role: "user", text });
    this.flushPendingInput();
  }

  noteToolResultInput(toolCallId: string, toolName: string | undefined, text: string): void {
    this.pendingInput.push({ role: "tool", text, toolCallId, toolName });
    this.flushPendingInput();
  }

  private llmEventGenAiSystem(): Record<string, string> | undefined {
    const ps = this.llm?.providerSystem;
    return ps ? { [this.providerNameKey()]: ps } : undefined;
  }

  private flushPendingInput(): void {
    if (!this.llm || this.pendingInput.length === 0) return;
    const emitEvents = this.semconv === "1.36";
    for (const m of this.pendingInput) {
      if (m.role === "user") {
        if (!this.shouldCapturePrompt()) continue;
        if (emitEvents) {
          const attrs: Record<string, unknown> = { role: "user", ...this.llmEventGenAiSystem() };
          attrs.content = clampAttr(m.text);
          this.llm.span.addEvent(EVENT_GEN_AI_USER_MESSAGE, attrs as Attributes);
        }
        this.llm.inputMessages.push({ role: "user", parts: [{ type: "text", content: m.text }] });
      } else if (this.shouldCaptureToolContent()) {
        if (emitEvents) {
          const attrs: Record<string, unknown> = {
            role: "tool",
            ...this.llmEventGenAiSystem(),
            [ATTR_GEN_AI_TOOL_CALL_ID]: m.toolCallId ?? "",
            ...(m.toolName ? { [ATTR_GEN_AI_TOOL_NAME]: m.toolName } : {}),
            content: clampAttr(m.text),
          };
          this.llm.span.addEvent(EVENT_GEN_AI_TOOL_MESSAGE, attrs as Attributes);
        }
        this.llm.inputMessages.push({
          role: "tool",
          parts: [{ type: "tool_call_response", id: m.toolCallId, name: m.toolName, response: m.text }],
        });
      }
    }
    this.pendingInput = [];
  }

  /** Record HTTP response details on the active LLM span; return true if retry detected. */
  recordProviderResponse(status: number, headers: Record<string, string>): boolean {
    if (!this.llm) return false;
    this.llm.attempts++;
    this.llm.httpStatus = status;
    this.llm.span.setAttribute(ATTR_HTTP_STATUS_CODE, status);
    // HTTP header names are case-insensitive; normalize so mixed-case maps still match.
    const hdr = lowercaseKeys(headers);
    const respId = hdr["x-request-id"] ?? hdr["request-id"] ?? hdr["anthropic-request-id"] ?? hdr["openai-response-id"];
    if (respId) this.llm.span.setAttribute(ATTR_GEN_AI_RESPONSE_ID, respId);
    const retry = this.llm.attempts > 1;
    if (retry) {
      try {
        const attrs: Attributes = this.commonAttrs();
        if (this.llm.requestModel) attrs[ATTR_GEN_AI_REQUEST_MODEL] = this.llm.requestModel;
        attrs[ATTR_HTTP_STATUS_CODE] = status;
        this.opts.metrics()?.providerRetries.add(1, attrs);
      } catch { /* noop */ }
    }
    if (status >= 400) {
      // Track the failure on the slot, never on the span. Span attributes
      // cannot be removed once set (setAttribute ignores null/undefined), so
      // stamping an attempt-scoped error here would survive a successful
      // retry and misreport the request. The final outcome is applied once
      // at span finalization (applyLlmErrorState).
      this.llm.failedAttempts++;
      this.llm.lastErrorType = categorizeHttpError(status);
      this.llm.lastErrorStatus = status;
    } else if (this.llm.lastErrorType !== undefined) {
      // A retry succeeded: the request as a whole did not fail.
      this.llm.lastErrorType = undefined;
      this.llm.lastErrorStatus = undefined;
    }
    return retry;
  }

  /**
   * Stamp the final provider outcome on an LLM span. Called once at span
   * finalization (completeLlm error path, endLlm, orphan sweep): a failed
   * attempt that a retry recovered leaves no markers, while a request whose
   * final attempt failed keeps its error.type and ERROR status.
   */
  private applyLlmErrorState(span: Span): void {
    const slot = this.llm;
    if (!slot?.lastErrorType) return;
    span.setAttribute(ATTR_ERROR_TYPE, slot.lastErrorType);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: slot.lastErrorStatus !== undefined ? `HTTP ${slot.lastErrorStatus}` : "provider error",
    });
  }

  /** Finalize the LLM span from the completed AssistantMessage. */
  completeLlm(message: MessageShapes.AssistantMessage): void {
    if (!this.llm) return;
    const llm = this.llm;
    if (message.responseModel) {
      llm.responseModel = message.responseModel;
      llm.span.setAttribute(ATTR_GEN_AI_RESPONSE_MODEL, message.responseModel);
    }
    if (message.responseId) llm.span.setAttribute(ATTR_GEN_AI_RESPONSE_ID, message.responseId);
    const finish = message.stopReason;
    if (finish) llm.span.setAttribute(ATTR_GEN_AI_RESPONSE_FINISH_REASONS, [finish]);
    this.applyUsageAttrs(message);
    this.accumulateSessionUsageFromMessage(message);
    this.emitAssistantMessageEvents(message);
    llm.toolCallCount = countToolCalls(message);
    this.recordLlmMetrics();
    if (finish === "error" || finish === "aborted") {
      llm.span.setAttribute(ATTR_PI_CANCELLED, finish === "aborted");
      if (message.errorMessage) {
        llm.span.setAttribute(ATTR_EXCEPTION_MESSAGE, message.errorMessage);
      }
      // Stamp the final attempt's HTTP category (when one was seen) before
      // the status below overrides the message with the provider's own text.
      this.applyLlmErrorState(llm.span);
      llm.span.setStatus({
        code: SpanStatusCode.ERROR,
        message: message.errorMessage ?? finish,
      });
      if (finish === "error") {
        // One session error per failed request. Retries that recovered do
        // not count (recordProviderResponse stopped counting per attempt).
        this.errorCount++;
      }
    }
    llm.span.end();
    this.llm = null;
  }

  endLlm(opts: { reason: string; cancelled?: boolean; error?: unknown } = { reason: "end" }): void {
    if (!this.llm) return;
    const llm = this.llm;
    if (opts.cancelled) llm.span.setAttribute(ATTR_PI_CANCELLED, true);
    else if (opts.reason !== "end") llm.span.setAttribute(ATTR_PI_ORPHANED, true);
    this.setStatusFromError(llm.span, opts.error);
    this.applyLlmErrorState(llm.span);
    llm.span.end();
    this.llm = null;
  }

  private applyUsageAttrs(m: MessageShapes.AssistantMessage): void {
    if (!this.llm) return;
    const u = m.usage;
    if (!u) return;
    const set = (k: string, v: unknown) => {
      if (typeof v === "number" && Number.isFinite(v)) this.llm!.span.setAttribute(k, v);
    };
    set(ATTR_GEN_AI_INPUT_TOKENS, u.input);
    set(ATTR_GEN_AI_OUTPUT_TOKENS, u.output);
    set(ATTR_GEN_AI_CACHE_READ_TOKENS, u.cacheRead);
    set(ATTR_GEN_AI_CACHE_WRITE_TOKENS, u.cacheWrite);
    set(ATTR_GEN_AI_REASONING_TOKENS, u.reasoning);
    if (typeof u.cacheWrite1h === "number") {
      set(ATTR_GEN_AI_CACHE_WRITE_1H_TOKENS, u.cacheWrite1h);
    }
    if (typeof u.cost?.total === "number") {
      set(ATTR_GEN_AI_COST_USD, u.cost.total);
    }
    this.llm.inputTokens = u.input;
    this.llm.outputTokens = u.output;
    this.llm.reasoningTokens = u.reasoning;
    this.llm.cacheReadTokens = u.cacheRead;
    this.llm.cacheWriteTokens = u.cacheWrite;
    this.llm.cacheWrite1hTokens = u.cacheWrite1h;
  }

  private recordLlmMetrics(): void {
    if (!this.llm) return;
    const m = this.opts.metrics();
    if (!m) return;
    const elapsedSec = Number(process.hrtime.bigint() - this.llm.startNs) / 1e9;
    const base: Attributes = this.commonAttrs();
    if (this.llm.requestModel) base[ATTR_GEN_AI_REQUEST_MODEL] = this.llm.requestModel;
    if (this.llm.responseModel) base[ATTR_GEN_AI_RESPONSE_MODEL] = this.llm.responseModel;
    try {
      m.opDuration.record(elapsedSec, base);
      // Input/output are always recorded when present (including zero). Cache
      // and reasoning types only when positive so every request does not emit
      // a stack of zero-valued series.
      const recordTokens = (value: number | undefined, tokenType: string, requirePositive = false) => {
        if (typeof value !== "number" || !Number.isFinite(value)) return;
        if (requirePositive && value <= 0) return;
        m.tokenUsage.record(value, { ...base, [ATTR_GEN_AI_TOKEN_TYPE]: tokenType });
      };
      recordTokens(this.llm.inputTokens, "input");
      recordTokens(this.llm.outputTokens, "output");
      recordTokens(this.llm.cacheReadTokens, "cache_read", true);
      recordTokens(this.llm.cacheWriteTokens, "cache_write", true);
      recordTokens(this.llm.cacheWrite1hTokens, "cache_write_1h", true);
      recordTokens(this.llm.reasoningTokens, "reasoning", true);
    } catch { /* best-effort */ }
  }

  private emitAssistantMessageEvents(m: MessageShapes.AssistantMessage): void {
    if (!this.llm) return;
    const text = extractAssistantText(m);
    const toolCalls = extractToolCalls(m, this.shouldCaptureToolContent());
    if (this.shouldCapturePrompt()) {
      // The legacy message/choice events duplicate the JSON message
      // attributes below, so the 1.37 dialect drops them: captured content
      // ships once, in the attributes current backends read.
      if (this.semconv === "1.36") {
        const asstAttrs: Record<string, unknown> = { role: "assistant", ...this.llmEventGenAiSystem() };
        if (text) asstAttrs.content = clampAttr(text);
        if (toolCalls.length) asstAttrs["tool_calls"] = clampAttr(toolCalls);
        this.llm.span.addEvent(EVENT_GEN_AI_ASSISTANT_MESSAGE, asstAttrs as Attributes);
        const finish = m.stopReason ?? "stop";
        // The choice event keeps only the decision metadata. An earlier
        // version embedded the full message JSON, a third copy of the
        // completion on the same span.
        this.llm.span.addEvent(EVENT_GEN_AI_CHOICE, {
          ...this.llmEventGenAiSystem(),
          index: 0,
          finish_reason: finish,
        });
      }
      // Aspire-style JSON message attributes (read by 9.x AI panel and others).
      if (this.llm.inputMessages.length > 0) {
        this.llm.span.setAttribute(ATTR_GEN_AI_INPUT_MESSAGES, clampAttr(this.llm.inputMessages));
      }
      const outputParts: Array<Record<string, unknown>> = [];
      if (text) outputParts.push({ type: "text", content: text });
      for (const tc of toolCalls) {
        outputParts.push({ type: "tool_call", id: tc.id, name: tc.function.name, arguments: tc.function.arguments });
      }
      this.llm.span.setAttribute(
        ATTR_GEN_AI_OUTPUT_MESSAGES,
        clampAttr([{ role: "assistant", parts: outputParts, finish_reason: m.stopReason ?? "stop" }]),
      );
    }
  }

  // ------------------------------------------------------------------ tools
  startTool(toolCallId: string, toolName: string, input: unknown): void {
    const parent = this.turn?.ctx ?? this.interaction?.ctx ?? otelContext.active();
    const attrs = this.commonAttrs();
    attrs[ATTR_GEN_AI_TOOL_NAME] = toolName;
    attrs[ATTR_GEN_AI_TOOL_CALL_ID] = toolCallId;
    if (this.shouldCaptureToolContent() && input !== undefined) {
      attrs[ATTR_GEN_AI_TOOL_CALL_ARGUMENTS] = clampAttr(input);
    }
    const spanOptions: { attributes: Attributes; links?: Array<{ context: SpanContext }> } = { attributes: attrs };
    if (this.llm) {
      // Only link when the LLM span has a real span context. When traces are
      // disabled the runtime hands the tracker a no-op tracer whose span
      // contexts carry empty trace/span ids; a link to such a context is
      // invalid per the OTel spec, so skip it.
      const ctx = this.llm.span.spanContext();
      if (ctx.traceId && ctx.spanId) {
        spanOptions.links = [{ context: ctx }];
      }
    }
    const span = this.opts.tracer.startSpan(spanToolName(toolName), spanOptions, parent);
    this.tools.set(toolCallId, {
      span,
      ctx: trace.setSpan(parent, span),
      name: toolName,
      startNs: process.hrtime.bigint(),
      startMs: this.now(),
    });
    this.interactionToolCount++;
    this.sessionToolCount++;
  }

  endTool(toolCallId: string, isError: boolean, result: unknown): void {
    const slot = this.tools.get(toolCallId);
    if (!slot) return;
    this.tools.delete(toolCallId);
    slot.span.setAttribute(ATTR_PI_TOOL_IS_ERROR, isError);
    if (this.shouldCaptureToolContent() && result !== undefined) {
      slot.span.setAttribute(ATTR_GEN_AI_TOOL_CALL_RESULT, clampAttr(result));
    }
    const elapsedMs = Number(process.hrtime.bigint() - slot.startNs) / 1e6;
    slot.span.setAttribute("pi.tool.duration_ms", elapsedMs);
    if (isError) {
      slot.span.setAttribute(ATTR_ERROR_TYPE, "tool_error");
      slot.span.setStatus({ code: SpanStatusCode.ERROR, message: "tool execution failed" });
    }
    slot.span.end();
    try {
      const attrs: Attributes = this.commonAttrs();
      attrs[ATTR_GEN_AI_TOOL_NAME] = slot.name;
      if (isError) attrs[ATTR_ERROR_TYPE] = "tool_error";
      this.opts.metrics()?.toolCalls.add(1, attrs);
    } catch { /* noop */ }
  }

  private endAllTools(reason: string): void {
    const markOrphaned = reason !== "end" && reason !== "session_end";
    for (const [, slot] of this.tools) {
      if (markOrphaned) slot.span.setAttribute(ATTR_PI_ORPHANED, true);
      slot.span.end();
    }
    this.tools.clear();
  }

  // --------------------------------------------------------------- helpers
  private accumulateSessionUsageFromMessage(m: MessageShapes.AssistantMessage): void {
    const u = m.usage;
    if (!u) return;
    if (typeof u.input === "number" && Number.isFinite(u.input)) this.totalInputTokens += u.input;
    if (typeof u.output === "number" && Number.isFinite(u.output)) this.totalOutputTokens += u.output;
    const cost = u.cost?.total;
    if (typeof cost === "number" && Number.isFinite(cost)) this.totalCostUsd += cost;
  }

  private commonAttrs(): Attributes {
    const attrs: Attributes = {
      [ATTR_PI_CWD]: this.opts.cwd,
    };
    const sid = this.opts.sessionId();
    if (sid) attrs[ATTR_PI_SESSION_ID] = sid;
    const file = this.opts.sessionFile();
    if (file) attrs[ATTR_PI_SESSION_FILE] = file;
    return attrs;
  }

  private setStatusFromError(span: Span, error: unknown): void {
    if (!error) return;
    const errName = error instanceof Error ? error.name : "Error";
    const errMsg = error instanceof Error ? error.message : String(error);
    const combined = `${errName} ${errMsg}`;
    span.setAttribute(ATTR_ERROR_TYPE, categorizeThrownError(combined, errName));
    span.setAttribute(ATTR_EXCEPTION_MESSAGE, errMsg);
    span.setStatus({ code: SpanStatusCode.ERROR, message: errMsg });
    // The end* methods pass the same error down the LLM -> turn -> interaction
    // cascade, so a single logical error reaches here up to three times.
    // Count each distinct error once toward the session's pi.error_count by
    // remembering the last reference we incremented on.
    if (this.lastCountedError !== error) {
      this.errorCount++;
      this.lastCountedError = error;
    }
  }

  /**
   * Mark all active spans as cancelled (Esc/abort).
   * Returns true when there was an in-flight turn to cancel, false otherwise
   * (e.g. the abort fired outside a turn). Callers use the return value to
   * decide whether to bump the turn-cancellation metric so it does not count
   * aborts with nothing to cancel.
   */
  markCancelled(): boolean {
    if (this.llm) this.llm.span.setAttribute(ATTR_PI_CANCELLED, true);
    for (const [, slot] of this.tools) slot.span.setAttribute(ATTR_PI_CANCELLED, true);
    const hadTurn = this.turn !== null;
    if (this.turn) this.turn.span.setAttribute(ATTR_PI_CANCELLED, true);
    if (this.interaction) this.interaction.span.setAttribute(ATTR_PI_CANCELLED, true);
    return hadTurn;
  }

  /** Surface the active interaction's trace id (e.g. for UI display). */
  activeTraceId(): string | undefined {
    return (this.session ?? this.interaction)?.span.spanContext().traceId;
  }

  /** Prompt/completion text is captured in full and no_tool_content modes. */
  private shouldCapturePrompt(): boolean {
    const c = this.opts.captureContent;
    return c === "no_tool_content" || c === "full";
  }

  /** Tool args/results are captured only in full mode. */
  private shouldCaptureToolContent(): boolean {
    return this.opts.captureContent === "full";
  }

  /** The dialect's provider identity attribute key. */
  private providerNameKey(): string {
    return this.semconv === "1.36" ? ATTR_GEN_AI_SYSTEM : ATTR_GEN_AI_PROVIDER_NAME;
  }
}

// ---------------------------------------------------------------------------
// Message-shape helpers (typed against @earendil-works/pi-ai)
// ---------------------------------------------------------------------------

function extractAssistantText(m: MessageShapes.AssistantMessage): string {
  const parts: string[] = [];
  for (const p of m.content) {
    if (p.type === "text") parts.push(p.text);
  }
  return parts.join("\n");
}

function extractToolCalls(
  m: MessageShapes.AssistantMessage,
  captureToolArgs: boolean,
): Array<{
  id: string;
  type: "function";
  function: { name: string; arguments?: string };
}> {
  const out: Array<{ id: string; type: "function"; function: { name: string; arguments?: string } }> = [];
  for (const p of m.content) {
    if (p.type !== "toolCall") continue;
    const tc = p;
    const fn: { name: string; arguments?: string } = { name: tc.name };
    if (captureToolArgs) {
      fn.arguments = typeof tc.arguments === "string" ? tc.arguments : clampAttr(tc.arguments);
    }
    out.push({ id: tc.id, type: "function", function: fn });
  }
  return out;
}

function countToolCalls(m: MessageShapes.AssistantMessage): number {
  let n = 0;
  for (const p of m.content) if (p.type === "toolCall") n++;
  return n;
}

/** Flatten any message's content to text. */
export function extractMessageText(message: { content?: unknown }): string {
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const p of content as Array<{ type?: string; text?: string }>) {
    if (p?.type === "text" && typeof p.text === "string") parts.push(p.text);
  }
  return parts.join("\n");
}

function prefixKeys(prefix: string, obj: Record<string, number | string>): Record<string, number | string> {
  const out: Record<string, number | string> = {};
  for (const [k, v] of Object.entries(obj)) out[`${prefix}.${k}`] = v;
  return out;
}

function lowercaseKeys(headers: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(headers)) out[k.toLowerCase()] = v;
  return out;
}

// ---------------------------------------------------------------------------
// Structural message shapes.
// These mirror @earendil-works/pi-ai's AssistantMessage / ToolCall / Usage.
// We declare them locally rather than importing from pi-ai (a transitive dep
// of pi-coding-agent that may not resolve from an extension's node_modules).
// Keeping them structural also lets us tolerate minor field additions across
// pi versions without a type error.
// ---------------------------------------------------------------------------
export namespace MessageShapes {
  export interface Usage {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
    cacheWrite1h?: number;
    /** Reasoning/thinking tokens reported by some providers (e.g. OpenAI o-series). */
    reasoning?: number;
    totalTokens?: number;
    cost?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number; total?: number };
  }
  export type StopReason = string; // "stop" | "end_turn" | "max_tokens" | "tool_use" | "error" | "aborted" | ...
  export interface TextPart { type: "text"; text: string }
  export interface ToolCallPart {
    type: "toolCall";
    id: string;
    name: string;
    arguments: Record<string, unknown> | string;
  }
  export type ContentPart = TextPart | ToolCallPart | { type: "thinking" } | { type: "other" };
  export interface AssistantMessage {
    role: "assistant";
    // `content` is a discriminated union on `type`. The last member is a
    // catch-all for parts we don't model (thinking, redacted_thought, ...).
    content: ContentPart[];
    model: string;
    responseModel?: string;
    responseId?: string;
    usage: Usage;
    stopReason: StopReason;
    errorMessage?: string;
  }
}

/** Default timer: setInterval with unref so the sweep never blocks process exit. */
function defaultSetTimer(fn: () => void, ms: number): () => void {
  const h = setInterval(fn, ms);
  h.unref?.();
  return () => clearInterval(h);
}
