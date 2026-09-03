import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { normalizeLogPayload } from "../src/index.ts";

/**
 * The pi-otel:log channel accepts payloads from other extensions, which are
 * untrusted. normalizeLogPayload is the validation boundary: it must reject
 * non-objects and payloads without a usable eventName, coerce severity to a
 * known value, and drop attribute values the OTLP log model cannot carry
 * (nested objects, arrays) while keeping scalars and clamping long strings.
 */
describe("normalizeLogPayload (pi-otel:log channel validation)", () => {
  test("accepts a well-formed payload", () => {
    const out = normalizeLogPayload({
      eventName: "my-ext.event",
      severity: "warn",
      body: "hello",
      attributes: { ok: true, n: 3, s: "x" },
    });
    assert.deepEqual(out, {
      eventName: "my-ext.event",
      severity: "warn",
      body: "hello",
      attributes: { ok: true, n: 3, s: "x" },
    });
  });

  test("rejects non-objects", () => {
    assert.equal(normalizeLogPayload(null), null);
    assert.equal(normalizeLogPayload(undefined), null);
    assert.equal(normalizeLogPayload("nope"), null);
    assert.equal(normalizeLogPayload(42), null);
    assert.equal(normalizeLogPayload([], ), null);
  });

  test("rejects payloads without a string eventName", () => {
    assert.equal(normalizeLogPayload({ severity: "info" }), null);
    assert.equal(normalizeLogPayload({ eventName: 42 }), null);
    assert.equal(normalizeLogPayload({ eventName: "" }), null);
  });

  test("coerces unknown severity to info", () => {
    const out = normalizeLogPayload({ eventName: "e", severity: "bogus" });
    assert.equal(out!.severity, "info");
  });

  test("defaults missing body to empty string", () => {
    const out = normalizeLogPayload({ eventName: "e" });
    assert.equal(out!.body, "");
  });

  test("drops non-scalar attribute values", () => {
    const out = normalizeLogPayload({
      eventName: "e",
      attributes: {
        keep_str: "s",
        keep_num: 1,
        keep_bool: true,
        drop_obj: { nested: "secret" },
        drop_arr: [1, 2, 3],
        drop_null: null,
        drop_undef: undefined,
      },
    });
    assert.deepEqual(out!.attributes, {
      keep_str: "s",
      keep_num: 1,
      keep_bool: true,
    });
    // Ensure the nested secret value is not present anywhere in the output.
    assert.ok(!JSON.stringify(out!.attributes).includes("secret"));
  });

  test("accepts attributes when missing entirely", () => {
    const out = normalizeLogPayload({ eventName: "e" });
    assert.deepEqual(out!.attributes, {});
  });

  test("clamps long string attribute values to the attribute ceiling", () => {
    const long = "a".repeat(100 * 1024);
    const out = normalizeLogPayload({ eventName: "e", attributes: { big: long } });
    const v = out!.attributes.big as string;
    assert.ok(v.length < long.length, "value was clamped");
    assert.ok(v.endsWith("…[truncated]"), "truncation marker present");
    assert.ok(Buffer.byteLength(v, "utf8") <= 64 * 1024, "within 64 KiB");
  });

  test("clamps an oversized eventName", () => {
    const long = "e".repeat(100 * 1024);
    const out = normalizeLogPayload({ eventName: long });
    const v = out!.eventName as string;
    assert.ok(v.length < long.length, "eventName was clamped");
    assert.ok(v.endsWith("…[truncated]"), "truncation marker present");
    assert.ok(Buffer.byteLength(v, "utf8") <= 64 * 1024, "within 64 KiB");
  });

  test("clamps a long body", () => {
    const long = "b".repeat(100 * 1024);
    const out = normalizeLogPayload({ eventName: "e", body: long });
    assert.ok(out!.body.length < long.length);
    assert.ok(out!.body.endsWith("…[truncated]"));
  });
});
