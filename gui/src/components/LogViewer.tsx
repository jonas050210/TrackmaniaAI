/** Scrollable log viewer with severity colouring. */

export function LogViewer({ lines, maxHeight = 420 }: { lines: string[]; maxHeight?: number }) {
  return (
    <div className="log" style={{ maxHeight }}>
      {lines.length === 0 && <span className="faint">no output yet</span>}
      {lines.map((line, index) => {
        const tone = line.includes("ERROR") || line.includes("FAILED")
          ? "error"
          : line.includes("WARNING") || line.includes("WARN")
            ? "warn"
            : "info";
        return (
          <div key={index} className={`line ${tone}`}>
            {line}
          </div>
        );
      })}
    </div>
  );
}
