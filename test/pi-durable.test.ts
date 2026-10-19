/**
 * Pi Durable compatibility. Drives a real Harness with the faux provider and
 * asserts the GenAI spans the durable extension emits.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { BasicTracerProvider, InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { Type, createModels, fauxAssistantMessage, fauxProvider, fauxToolCall } from "@earendil-works/pi-ai";
import { Harness, MemoryStorage, createRegistry, defineExtension, defineTool } from "@earendil-works/pi-durable";
import { conversationTraceId, createDurableTraces } from "../src/pi-durable.ts";

const context = BACKGROUND_CONTEXT;

async function run(script: ReturnType<typeof fauxAssistantMessage>[]) {
  const exporter = new InMemorySpanExporter();
  const provider = new BasicTracerProvider({
    spanProcessors: [new (await import("@opentelemetry/sdk-trace-base")).SimpleSpanProcessor(exporter)],
  });
  const tracer = provider.getTracer("pi-otel-durable-test");
  const traces = createDurableTraces({ tracer, captureContent: "metadata_only" });

  const echo = defineTool({
    name: "echo",
    description: "Returns its text",
    parameters: Type.Object({ text: Type.String() }),
    replay: "safe",
    execute: async (args) => ({ output: [{ type: "text", text: args.text }] }),
  });
  const registry = createRegistry();
  registry.install(defineExtension({ name: "app", tools: [echo] }));
  registry.install(traces.extension);

  const faux = fauxProvider({ provider: "faux", models: [{ id: "faux-1" }] });
  faux.setResponses(script);
  const models = createModels();
  models.setProvider(faux.provider);

  const harness = await Harness.open(new MemoryStorage(), { models, registry }, context);
  try {
    const root = await harness.root(context, {
      agent: { model: { provider: "faux", modelId: "faux-1" } },
    });
    const submission = await root.submit({ type: "input", content: "say hi", requestId: "job-1" }, context);
    const settled = await submission.wait(context);
    await provider.forceFlush();
    return { settled, spans: exporter.getFinishedSpans(), conversationId: String(root.id) };
  } finally {
    traces.endConversation("ignored");
    await harness.close(context);
    await provider.shutdown();
  }
}

test("conversationTraceId is stable and 32 hex chars", () => {
  const id = conversationTraceId("42");
  assert.equal(id, conversationTraceId("42"));
  assert.notEqual(id, conversationTraceId("43"));
  assert.match(id, /^[0-9a-f]{32}$/);
});

test("a durable turn emits a chat span and a linked tool span on one trace", async () => {
  const { spans, conversationId } = await run([
    fauxAssistantMessage([fauxToolCall("echo", { text: "hi" })], { stopReason: "toolUse" }),
    fauxAssistantMessage("done"),
  ]);
  const names = spans.map((span) => span.name).sort();
  assert.deepEqual(names, ["pi.llm_request", "pi.llm_request", "pi.tool.echo"].sort());

  const expected = conversationTraceId(conversationId);
  for (const span of spans) {
    assert.equal(span.spanContext().traceId, expected, span.name);
    assert.equal(span.attributes["gen_ai.conversation.id"], conversationId);
    assert.equal(span.attributes["gen_ai.agent.name"], "pi");
  }

  const chats = spans.filter((span) => span.name === "pi.llm_request");
  const finished = chats.find((span) => JSON.stringify(span.attributes["gen_ai.response.finish_reasons"]).includes("stop"));
  assert.ok(finished, "a chat span records the final stop reason");
  assert.equal(finished.attributes["gen_ai.operation.name"], "chat");
  assert.equal(finished.attributes["gen_ai.provider.name"], "faux");
  assert.equal(finished.attributes["gen_ai.request.model"], "faux-1");
  assert.equal(finished.attributes["gen_ai.input.messages"], undefined);

  const tool = spans.find((span) => span.name === "pi.tool.echo");
  assert.ok(tool);
  assert.equal(tool.attributes["gen_ai.operation.name"], "execute_tool");
  assert.equal(tool.attributes["gen_ai.tool.name"], "echo");
  assert.equal(tool.attributes["gen_ai.tool.call.arguments"], undefined);
  assert.ok(tool.links.length >= 1, "the tool span links to the llm span");
  assert.equal(tool.links[0]?.context.traceId, expected);
});

test("metadata_only stays clear of the prompt when the model errors", async () => {
  const { spans } = await run([
    fauxAssistantMessage("nope", { stopReason: "error", errorMessage: "provider down" }),
  ]);
  const chat = spans.find((span) => span.name === "pi.llm_request");
  assert.ok(chat);
  assert.equal(chat.status.code, 2);
  assert.equal(chat.attributes["error.type"], "error");
  assert.equal(chat.attributes["gen_ai.output.messages"], undefined);
});
