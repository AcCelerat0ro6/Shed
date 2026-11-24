import { Loader, truncateToWidth } from "@earendil-works/pi-tui";
import { theme } from "../theme/theme.js";

/**
 * 基础状态指示器，继承自 Loader 并支持在组件边框内联渲染旋转动画。
 */
export class StatusIndicator extends Loader {
  public kind: string;

  constructor(
    kind: string,
    ui: any,
    spinnerColorFn: (spinner: string) => string,
    messageColorFn: (msg: string) => string,
    message: string,
    indicator?: any,
  ) {
    super(ui, spinnerColorFn, messageColorFn, message, indicator);
    this.kind = kind;
    // 对内部定时器调用 unref，避免阻塞进程退出
    const timer = (this as any).intervalId;
    if (timer && typeof timer.unref === "function") {
      timer.unref();
    }
  }

  public override start(): void {
    super.start();
    const timer = (this as any).intervalId;
    if (timer && typeof timer.unref === "function") {
      timer.unref();
    }
  }

  public renderInBorder(width: number): string {
    const lines = super.render(width + 2);
    const line = lines[1] ?? lines[0] ?? "";
    const clean = line.startsWith(" ")
      ? line.slice(1).trimEnd()
      : line.trimEnd();
    return truncateToWidth(clean, width, "");
  }

  public renderSpinnerInBorder(width: number): string {
    const ind = (this as any).getRenderedIndicator?.() ?? "⠋";
    return truncateToWidth(ind, width, "");
  }

  public dispose(): void {
    this.stop();
  }
}

/**
 * 运行中状态指示器，可嵌入 CustomEditor 顶部边框展示实时执行状态。
 */
export class WorkingStatusIndicator extends StatusIndicator {
  constructor(
    ui: any,
    message = "Working",
    indicator?: any,
    colorFn?: (text: string) => string,
  ) {
    super(
      "working",
      ui,
      colorFn ?? ((text: string) => theme.fg("accent", text)),
      colorFn ?? ((text: string) => theme.fg("muted", text)),
      message,
      indicator,
    );
  }
}

/**
 * 上下文压缩状态指示器，区分手动触发与上下文溢出自动触发场景。
 */
export class CompactionStatusIndicator extends StatusIndicator {
  constructor(ui: any, reason: "manual" | "overflow" = "manual") {
    const cancelHint = "(Esc to cancel)";
    const label =
      reason === "manual"
        ? `Compacting context... ${cancelHint}`
        : "Context overflow, auto-compacting... (Esc to cancel)";
    super(
      "compaction",
      ui,
      (spinner: string) => theme.fg("accent", spinner),
      (text: string) => theme.fg("muted", text),
      label,
    );
  }
}
