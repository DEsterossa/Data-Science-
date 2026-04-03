import { type FormEvent, useState } from 'react'

import { predict } from './api/client'
import type { PredictionResponse } from './api/types'
import { CategoryBars } from './components/CategoryBars'

export default function App() {
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [emptyHint, setEmptyHint] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)

    const trimmed = description.trim()
    if (!trimmed) {
      setEmptyHint(true)
      setResult(null)
      return
    }

    setEmptyHint(false)
    setLoading(true)
    setResult(null)

    try {
      const data = await predict(trimmed)
      setResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Не удалось выполнить запрос'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900">
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 py-10">
          <p className="text-xs font-semibold uppercase tracking-widest text-indigo-600">
            ML demo
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-900 sm:text-4xl">
            Автокатегоризация товаров
          </h1>
          <p className="mt-3 max-w-2xl text-base leading-relaxed text-zinc-600">
            Введите текст описания товара или объявления <span className="font-medium text-zinc-800">на английском языке</span>
            {' '}(модель и векторизация рассчитаны на English). Сервис вернёт наиболее вероятную категорию,
            уверенность модели, задержку ответа и распределение по всем классам.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-4 py-10">
        <form
          onSubmit={handleSubmit}
          className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm"
          noValidate
        >
          <label htmlFor="description" className="block text-sm font-medium text-zinc-800">
            Описание{' '}
            <span className="font-normal text-amber-800">(только английский текст)</span>
          </label>
          <textarea
            id="description"
            name="description"
            rows={6}
            value={description}
            onChange={(event) => {
              setDescription(event.target.value)
              if (emptyHint && event.target.value.trim()) {
                setEmptyHint(false)
              }
            }}
            placeholder="e.g. Noise cancelling over-ear headphones with charging case"
            className="mt-2 w-full resize-y rounded-xl border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-inner outline-none ring-indigo-500 transition focus:border-indigo-500 focus:ring-2"
            aria-invalid={emptyHint}
            aria-describedby={emptyHint ? 'description-empty' : undefined}
          />
          {emptyHint ? (
            <p id="description-empty" className="mt-2 text-sm text-red-600" role="alert">
              Введите непустое описание, чтобы получить предсказание.
            </p>
          ) : (
            <p className="mt-2 text-sm text-zinc-500">
              Пишите на английском: используются английские стоп-слова и обучение на англоязычных данных.
              Минимум один значимый символ; пробелы по краям обрезаются.
            </p>
          )}

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center justify-center rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-60"
              aria-busy={loading}
            >
              {loading ? 'Анализ...' : 'Классифицировать'}
            </button>
            {loading ? (
              <span className="text-sm text-zinc-500">Запрос к API выполняется</span>
            ) : null}
          </div>
        </form>

        {error ? (
          <div
            className="mt-8 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
            role="alert"
          >
            {error}
          </div>
        ) : null}

        {result ? (
          <section
            className="mt-10 space-y-8 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm"
            aria-live="polite"
          >
            <div>
              <h2 className="text-lg font-semibold text-zinc-900">Лучшая категория</h2>
              <p className="mt-3 text-3xl font-semibold tracking-tight text-indigo-700">
                {result.best_category}
              </p>
            </div>

            <dl className="grid gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-zinc-100 bg-zinc-50 px-4 py-3">
                <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                  Уверенность
                </dt>
                <dd className="mt-1 text-2xl font-semibold tabular-nums text-zinc-900">
                  {(result.best_confidence * 100).toFixed(1)}%
                </dd>
              </div>
              <div className="rounded-xl border border-zinc-100 bg-zinc-50 px-4 py-3">
                <dt className="text-xs font-medium uppercase tracking-wide text-zinc-500">
                  Задержка ответа
                </dt>
                <dd className="mt-1 text-2xl font-semibold tabular-nums text-zinc-900">
                  {result.latency_ms.toFixed(1)} ms
                </dd>
              </div>
            </dl>

            <div>
              <h3 className="text-base font-semibold text-zinc-900">Все категории</h3>
              <p className="mt-1 text-sm text-zinc-600">
                Нормализованные вероятности по классам (модель: TF-IDF + Logistic Regression).
              </p>
              <div className="mt-4">
                <CategoryBars predictions={result.predictions} bestCategory={result.best_category} />
              </div>
            </div>
          </section>
        ) : null}
      </main>

      <footer className="border-t border-zinc-200 bg-white py-8 text-center text-xs text-zinc-500">
        Учебный full-stack demo: React + FastAPI + Docker
      </footer>
    </div>
  )
}
