import test from "node:test";
import assert from "node:assert/strict";
import { LoginSelectorComponent } from "../dist/components/login-selector.js";

test("LoginSelectorComponent renders provider list in phase 1", () => {
  let submittedProvider = null;
  let submittedKey = null;
  let cancelled = false;

  const selector = new LoginSelectorComponent(
    (prov, key) => {
      submittedProvider = prov;
      submittedKey = key;
    },
    () => {
      cancelled = true;
    },
  );

  assert.ok(selector);
  const lines = selector.render(80);
  const text = lines.join("\n");

  assert.ok(text.includes("Login"));
  assert.ok(text.includes("auth.json"));
  assert.ok(text.includes("DeepSeek"));
  assert.ok(text.includes("OpenAI"));
  assert.ok(text.includes("Enter to select"));
});

test("LoginSelectorComponent transitions to phase 2 and submits key", () => {
  let submittedProvider = null;
  let submittedKey = null;
  let cancelled = false;

  const selector = new LoginSelectorComponent(
    (prov, key) => {
      submittedProvider = prov;
      submittedKey = key;
    },
    () => {
      cancelled = true;
    },
  );

  // 1. 按回车选中第一项 DeepSeek
  selector.handleInput("\r");

  // 应进入第二步：输入 API Key
  const linesPhase2 = selector.render(80);
  const textPhase2 = linesPhase2.join("\n");
  assert.ok(textPhase2.includes("DeepSeek"));
  assert.ok(textPhase2.includes("API Key"));

  // 2. 输入密钥并按回车提交
  for (const ch of "sk-test-deepseek-123") {
    selector.handleInput(ch);
  }
  selector.handleInput("\r");

  assert.equal(submittedProvider, "deepseek");
  assert.equal(submittedKey, "sk-test-deepseek-123");
});

test("LoginSelectorComponent shows auth.json location explanation for Antigravity and submits on Enter", () => {
  let submittedProvider = null;
  let submittedKey = null;

  const selector = new LoginSelectorComponent(
    (prov, key) => {
      submittedProvider = prov;
      submittedKey = key;
    },
    () => {},
  );

  // 输入 antigravity 过滤并回车选中
  for (const ch of "antigravity") {
    selector.handleInput(ch);
  }
  selector.handleInput("\r");

  // 应进入第二步并展示 auth.json 凭据路径说明
  const linesPhase2 = selector.render(80);
  const textPhase2 = linesPhase2.join("\n");
  assert.ok(textPhase2.includes("auth.json"));
  assert.ok(textPhase2.includes("~/.my-pi-agent/auth.json"));
  assert.ok(textPhase2.includes("~/.pi/agent/auth.json"));

  // 留空直接回车，触发自动从 auth.json 加载凭据
  selector.handleInput("\r");
  assert.equal(submittedProvider, "antigravity");
  assert.equal(submittedKey, "");
});
