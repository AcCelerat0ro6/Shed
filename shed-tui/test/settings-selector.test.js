import test from "node:test";
import assert from "node:assert/strict";
import { SettingsSelectorComponent } from "../dist/components/settings-selector.js";

test("SettingsSelectorComponent renders setting options", () => {
  const selector = new SettingsSelectorComponent(
    {
      auto_compact: true,
      default_model: "deepseek-chat",
      default_thinking_level: "medium",
      default_permission_mode: "review",
      theme: "dark",
    },
    () => {},
    () => {},
  );

  assert.ok(selector);
  const lines = selector.render(80);
  const text = lines.join("\n");

  assert.ok(text.includes("Settings"));
  assert.ok(text.includes("Auto Compaction"));
  assert.ok(text.includes("[x]")); // 已启用状态标记
  assert.ok(text.includes("deepseek-chat"));
  assert.ok(text.includes("medium"));
  assert.ok(text.includes("review"));
});

test("SettingsSelectorComponent handles toggling boolean and cycling options", () => {
  let updatedKey = null;
  let updatedVal = null;
  let closed = false;

  const customDefs = [
    {
      key: "auto_compact",
      label: "Auto Compaction",
      type: "boolean",
      description: "auto compact",
    },
    {
      key: "default_permission_mode",
      label: "Permission Mode",
      type: "cycle",
      options: ["review", "yolo", "strict"],
      description: "mode",
    },
  ];

  const selector = new SettingsSelectorComponent(
    {
      auto_compact: true,
      default_permission_mode: "review",
    },
    (key, val) => {
      updatedKey = key;
      updatedVal = val;
    },
    () => {
      closed = true;
    },
    customDefs,
  );

  // 1. 对首项（auto_compact）按回车切换为 false
  selector.handleInput("\r");
  assert.equal(updatedKey, "auto_compact");
  assert.equal(updatedVal, false);

  // 2. 向下移动至 default_permission_mode，按回车循环切换选项
  selector.handleInput("\x1b[B"); // 向下键
  selector.handleInput("\r"); // 从 review 切换为 yolo
  assert.equal(updatedKey, "default_permission_mode");
  assert.equal(updatedVal, "yolo");

  // 3. 按 Esc 关闭设置面板
  selector.handleInput("\x1b");
  assert.equal(closed, true);
});
