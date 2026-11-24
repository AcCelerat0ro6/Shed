import { matchesKey } from "@earendil-works/pi-tui";

/**
 * 统一回车键判断，兼容 VT100/ANSI（\r、\n）、Windows ConPTY/SSH（\r\n）以及底层按键匹配器的 return/enter 事件。
 */
export function isEnterKey(data: string): boolean {
   return (
      matchesKey(data, "return") ||
      matchesKey(data, "enter") ||
      data === "\r" ||
      data === "\n" ||
      data === "\r\n"
   );
}
