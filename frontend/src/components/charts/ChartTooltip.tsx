export default function ChartTooltip({ active, payload, label, formatter }: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string | number;
  formatter?: (name: string, value: number) => string;
}) {
  if (!active || !payload?.length) return null;

  return (
    <div className="bg-[#1a1a2e] border border-[#2a2a40] rounded-lg px-3 py-2 shadow-xl">
      {label !== undefined && (
        <p className="text-xs text-gray-400 mb-1 font-mono">{label}</p>
      )}
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
          <span className="text-gray-300">{entry.name}:</span>
          <span className="font-mono text-white">
            {formatter ? formatter(entry.name, entry.value) : entry.value.toExponential(2)}
          </span>
        </div>
      ))}
    </div>
  );
}
