'use client';

import { PHYSICS_MODES, type PhysicsMode } from '@/lib/constants';

interface Props {
  selected: PhysicsMode[];
  onChange: (modes: PhysicsMode[]) => void;
}

export default function PhysicsModeSelector({ selected, onChange }: Props) {
  const toggle = (mode: PhysicsMode) => {
    if (selected.includes(mode)) {
      if (selected.length > 1) onChange(selected.filter((m) => m !== mode));
    } else {
      onChange([...selected, mode]);
    }
  };

  return (
    <div className="flex gap-2 flex-wrap">
      {PHYSICS_MODES.map((m) => {
        const isActive = selected.includes(m.id);
        return (
          <button
            key={m.id}
            onClick={() => toggle(m.id)}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium border transition-all ${
              isActive
                ? 'border-opacity-50 text-white'
                : 'border-[#2a2a40] text-gray-500 hover:text-gray-300'
            }`}
            style={isActive ? { borderColor: m.color, backgroundColor: `${m.color}20`, color: m.color } : {}}
          >
            {m.label}
          </button>
        );
      })}
    </div>
  );
}
