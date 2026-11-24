import test from "node:test";
import assert from "node:assert/strict";
import { ThemeSelectorComponent } from "../dist/components/theme-selector.js";

const AVAILABLE_THEMES = ["dark", "light"];

test("ThemeSelectorComponent renders theme options with current indicator", () => {
  const selector = new ThemeSelectorComponent(
    "dark",
    AVAILABLE_THEMES,
    () => {},
    () => {},
  );

  assert.ok(selector);
  const lines = selector.render(80);
  const text = lines.join("\n");

  assert.ok(text.includes("dark"));
  assert.ok(text.includes("light"));
  assert.ok(text.includes("(current)"));
});

test("ThemeSelectorComponent dispatches preview on change, commits on Enter, reverts on Esc", () => {
  let previewed = null;
  let committed = null;
  let cancelled = false;

  const selector = new ThemeSelectorComponent(
    "dark",
    AVAILABLE_THEMES,
    (t) => {
      committed = t;
    },
    () => {
      cancelled = true;
    },
    (t) => {
      previewed = t;
    },
  );

  // 向下移动至 light 并触发实时预览
  selector.handleInput("\x1b[B"); // 向下键
  assert.equal(previewed, "light");

  // 按回车确认应用
  selector.handleInput("\r");
  assert.equal(committed, "light");

  // 按 Esc 取消并回滚
  selector.handleInput("\x1b");
  assert.equal(cancelled, true);
});
