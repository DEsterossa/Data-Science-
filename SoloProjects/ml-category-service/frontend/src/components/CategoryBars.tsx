import type { CategoryPrediction } from '../api/types'

interface CategoryBarsProps {
  predictions: CategoryPrediction[]
  bestCategory: string
}

export function CategoryBars({ predictions, bestCategory }: CategoryBarsProps) {
  const sorted = [...predictions].sort((a, b) => b.confidence - a.confidence)

  return (
    <ul className="space-y-3" aria-label="Вероятности по категориям">
      {sorted.map((item) => {
        const pct = Math.min(100, Math.max(0, item.confidence * 100))
        const isBest = item.category === bestCategory

        return (
          <li key={item.category}>
            <div className="flex items-center justify-between gap-3 text-sm">
              <span
                className={
                  isBest
                    ? 'font-semibold text-zinc-900'
                    : 'font-medium text-zinc-600'
                }
              >
                {item.category}
              </span>
              <span className="tabular-nums text-zinc-500">
                {(item.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div
              className="mt-1 h-2 overflow-hidden rounded-full bg-zinc-200"
              role="presentation"
            >
              <div
                className={
                  isBest
                    ? 'h-full rounded-full bg-indigo-600 transition-[width] duration-300'
                    : 'h-full rounded-full bg-zinc-400 transition-[width] duration-300'
                }
                style={{ width: `${pct}%` }}
              />
            </div>
          </li>
        )
      })}
    </ul>
  )
}
