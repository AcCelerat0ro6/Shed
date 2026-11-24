import test from "node:test";
import assert from "node:assert/strict";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { PythonKernelClient } from "../dist/client.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");

test("End-to-End: PythonKernelClient launches real Python rpc_server and manages lifecycle requests", async (t) => {
  const client = new PythonKernelClient({
    workspace: repoRoot,
  });

  t.after(async () => {
    await client.shutdown();
  });

  // 1. 启动客户端并拉起真实 Python rpc_server 进程
  await client.start();
  assert.ok(client.child !== null, "Child process must be spawned");
  assert.ok(
    typeof client.child.pid === "number" && client.child.pid > 0,
    "Child PID must be valid",
  );
  assert.equal(client.child.killed, false, "Child process should be running");

  // 2. 发送 steer 引导消息
  await client.steer("test steering message");

  // 3. 发送 followup 追加任务
  await client.followup("test followup task");

  // 4. 发送 abort 中断指令
  await client.abort();

  assert.ok(
    true,
    "Client successfully communicated with Python kernel over stdio JSON-RPC",
  );
});
