import { Box, Markdown, Spacer, Text } from "@earendil-works/pi-tui";
import { getMarkdownTheme, theme } from "../theme/theme.js";

export interface CompactionSummaryData {
  summary: string;
  tokensBefore: number;
}

/**
 * 上下文压缩摘要折叠卡片组件。
 * 默认以折叠态展示压缩前 Token 数，按 Ctrl+O 展开查看完整 Markdown 摘要。
 */
export class CompactionSummaryMessageComponent extends Box {
  private expanded = false;
  private message: CompactionSummaryData;

  constructor(message: CompactionSummaryData) {
    super(1, 1, (t: string) => theme.bg("customMessageBg", t));
    this.message = message;
    this.updateDisplay();
  }

  public setExpanded(expanded: boolean): void {
    this.expanded = expanded;
    this.updateDisplay();
  }

  public toggleExpanded(): void {
    this.expanded = !this.expanded;
    this.updateDisplay();
  }

  public override invalidate(): void {
    super.invalidate();
    this.updateDisplay();
  }

  private updateDisplay(): void {
    this.clear();
    const tokenStr = (this.message.tokensBefore ?? 0).toLocaleString();
    const label = theme.fg("customMessageLabel", "\x1b[1m[compaction]\x1b[22m");
    this.addChild(new Text(label, 0, 0));
    this.addChild(new Spacer(1));

    if (this.expanded) {
      const header = `**Compacted from ${tokenStr} tokens**\n\n`;
      this.addChild(
        new Markdown(header + this.message.summary, 0, 0, getMarkdownTheme(), {
          color: (text: string) => theme.fg("customMessageText", text),
        }),
      );
    } else {
      this.addChild(
        new Text(
          theme.fg("customMessageText", `Compacted from ${tokenStr} tokens (`) +
            theme.fg("dim", "Ctrl+O") +
            theme.fg("customMessageText", " to expand)"),
          0,
          0,
        ),
      );
    }
  }
}
