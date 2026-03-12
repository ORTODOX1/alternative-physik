'use client';

interface ParameterSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
  formatValue?: (v: number) => string;
}

export default function ParameterSlider({
  label, value, min, max, step, unit = '', onChange, formatValue,
}: ParameterSliderProps) {
  const display = formatValue ? formatValue(value) : `${value}`;

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between items-center">
        <label className="text-sm text-gray-300">{label}</label>
        <span className="text-sm font-mono text-blue-400">
          {display}{unit && <span className="text-gray-500 ml-1">{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
      />
    </div>
  );
}
