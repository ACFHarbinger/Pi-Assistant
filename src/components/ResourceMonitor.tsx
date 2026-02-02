import { useEffect, useMemo } from "react";
import { useResourceStore, ResourceSnapshot } from "../stores/resourceStore";

// Initialize the socket listener once in the main App component
export function useResourceMonitorInit() {
  const initResourceSocket = useResourceStore((s) => s.initResourceSocket);

  useEffect(() => {
    initResourceSocket();
  }, []);
}

export function ResourceMonitor() {
  const current = useResourceStore((s) => s.current);
  const history = useResourceStore((s) => s.history);

  if (!current) {
    return (
      <div className="p-4 text-white/40 text-center">
        Waiting for resource data...
      </div>
    );
  }

  return (
    <div className="p-4 space-y-6 animate-in fade-in duration-300">
      <h2 className="text-lg font-semibold text-white/90">System Resources</h2>

      {/* CPU */}
      <MetricCard
        label="CPU Usage"
        value={current.cpu_usage_percent}
        unit="%"
        history={history.map((s) => s.cpu_usage_percent)}
        color="#22c55e"
      />

      {/* Memory */}
      <MetricCard
        label="Memory"
        value={current.memory.percent}
        unit="%"
        subtext={`${formatBytes(current.memory.used)} / ${formatBytes(current.memory.total)}`}
        history={history.map((s) => s.memory.percent)}
        color="#3b82f6"
      />

      {/* Swap */}
      {current.swap.total > 0 && (
        <MetricCard
          label="Swap"
          value={current.swap.percent}
          unit="%"
          subtext={`${formatBytes(current.swap.used)} / ${formatBytes(current.swap.total)}`}
          history={history.map((s) => s.swap.percent)}
          color="#eab308"
        />
      )}

      {/* GPU(s) */}
      {current.gpu && Object.keys(current.gpu).length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-white/60">GPU Memory</h3>
          {Object.entries(current.gpu).map(([device, mem]) => (
            <MetricCard
              key={device}
              label={device}
              value={mem.total_mb > 0 ? (mem.used_mb / mem.total_mb) * 100 : 0}
              unit="%"
              subtext={`${mem.used_mb} MB / ${mem.total_mb} MB`}
              history={history.map((s) => {
                const g = s.gpu?.[device];
                return g && g.total_mb > 0 ? (g.used_mb / g.total_mb) * 100 : 0;
              })}
              color="#a855f7"
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  unit,
  subtext,
  history,
  color,
}: {
  label: string;
  value: number;
  unit: string;
  subtext?: string;
  history: number[];
  color: string;
}) {
  return (
    <div className="rounded-xl bg-white/5 border border-white/5 p-4 space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm text-white/70">{label}</span>
        <span className="text-lg font-mono" style={{ color }}>
          {value.toFixed(1)}
          {unit}
        </span>
      </div>
      {subtext && <p className="text-xs text-white/40">{subtext}</p>}
      <Sparkline data={history} color={color} />
    </div>
  );
}

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const height = 32;
  const width = 200;

  const points = useMemo(() => {
    if (data.length === 0) return "";
    const max = Math.max(...data, 100);
    const stepX = width / Math.max(data.length - 1, 1);
    return data
      .map((v, i) => {
        const x = i * stepX;
        const y = height - (v / max) * height;
        return `${x},${y}`;
      })
      .join(" ");
  }, [data]);

  return (
    <svg
      className="w-full"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
    >
      <polyline
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}

// Format bytes as human-readable
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
