import test from "node:test";
import assert from "node:assert/strict";
import { stripVTControlCharacters } from "node:util";
import { SessionSelectorComponent } from "../dist/components/session-selector.js";

const TEST_SESSIONS = [
  {
    id: "sess-1",
    name: "重构路径",
    path: "/test/sess-1.jsonl",
    cwd: "/test/workspace",
    modified: Math.floor(Date.now() / 1000) - 120, // 2 分钟前
    message_count: 5,
  },
  {
    id: "sess-2",
    name: "实现MCP支持",
    path: "/test/sess-2.jsonl",
    cwd: "/test/workspace",
    modified: Math.floor(Date.now() / 1000) - 3600, // 1 小时前
    message_count: 12,
  },
];

test("SessionSelectorComponent renders session items and metadata", async () => {
  let selected = null;
  let cancelled = false;

  const selector = new SessionSelectorComponent(
    async () => TEST_SESSIONS,
    (s) => {
      selected = s;
    },
    () => {
      cancelled = true;
    },
    undefined,
    "sess-1",
  );

  // 等待初始列表异步加载完成
  await new Promise((r) => setTimeout(r, 20));

  const lines = selector.render(80);
  const text = lines.join("\n");

  assert.ok(text.includes("Resume Session"));
  assert.ok(text.includes("重构路径"));
  assert.ok(text.includes("实现MCP支持"));
  assert.ok(text.includes("5 "));
});

test("SessionSelectorComponent renders forked child session with tree branch prefix (├─ / └─)", async () => {
  const SESSIONS_WITH_FORK = [
    {
      id: "main-sess",
      name: "main",
      path: "/test/main.jsonl",
      cwd: "/test/workspace",
      modified: Math.floor(Date.now() / 1000) - 86400,
      message_count: 8671,
    },
    {
      id: "forked-sess",
      name: "我接下来应该开发哪部分",
      path: "/test/forked.jsonl",
      parent_session: "main-sess",
      cwd: "/test/workspace",
      modified: Math.floor(Date.now() / 1000) - 3600,
      message_count: 1171,
    },
  ];

  const selector = new SessionSelectorComponent(
    async () => SESSIONS_WITH_FORK,
    () => {},
    () => {},
  );

  await new Promise((r) => setTimeout(r, 20));

  const lines = selector.render(100);
  const text = stripVTControlCharacters(lines.join("\n"));

  assert.ok(text.includes("main"));
  assert.ok(text.includes("8671"));
  assert.ok(
    text.includes("└─ 我接下来应该开发哪部分") ||
      text.includes("├─ 我接下来应该开发哪部分"),
  );
  assert.ok(text.includes("1171"));
});

test("SessionSelectorComponent handles selection and Tab scope toggle", async () => {
  let selected = null;
  let cancelled = false;
  let scopeArg = null;

  const selector = new SessionSelectorComponent(
    async (all) => {
      scopeArg = all;
      return TEST_SESSIONS;
    },
    (s) => {
      selected = s;
    },
    () => {
      cancelled = true;
    },
  );

  await new Promise((r) => setTimeout(r, 20));

  // 按向下键
  selector.handleInput("\x1b[B");

  // 按回车确认
  selector.handleInput("\r");
  assert.ok(selected);
  assert.equal(selected.id, "sess-2");

  // 按 Tab 切换会话筛选范围
  selector.handleInput("\t");
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(scopeArg, true);
});

test("SessionSelectorComponent supports Ctrl+D delete confirmation and cancel", async () => {
  let deleted = null;

  const selector = new SessionSelectorComponent(
    async () => [...TEST_SESSIONS],
    () => {},
    () => {},
    undefined,
    "sess-1", // sess-1 为当前活跃会话
    async (s) => {
      deleted = s;
    },
  );

  await new Promise((r) => setTimeout(r, 20));

  // 1. 尝试删除当前活跃会话（sess-1）应提示错误
  selector.handleInput("\x04"); // Ctrl+D
  const linesActiveErr = selector.render(80).join("\n");
  assert.ok(
    linesActiveErr.includes("Cannot delete the currently active session"),
  );
  assert.equal(deleted, null);

  // 2. 向下移动至非活跃会话 sess-2
  selector.handleInput("\x1b[B");

  // 3. 对 sess-2 按 Ctrl+D 进入删除确认态
  selector.handleInput("\x04"); // Ctrl+D
  const linesConfirm = selector.render(80).join("\n");
  assert.ok(linesConfirm.includes("Delete session?"));
  assert.ok(linesConfirm.includes("[delete?]"));

  // 4. 按 Esc 取消删除确认
  selector.handleInput("\x1b");
  const linesCancelled = selector.render(80).join("\n");
  assert.ok(!linesCancelled.includes("Delete session?"));
  assert.equal(deleted, null);

  // 5. 再次按 Ctrl+D 并按回车确认删除
  selector.handleInput("\x04"); // Ctrl+D
  selector.handleInput("\r"); // 回车确认
  await new Promise((r) => setTimeout(r, 20));
  assert.ok(deleted);
  assert.equal(deleted.id, "sess-2");
});
