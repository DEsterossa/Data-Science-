import { type FormEvent, useState } from 'react'

import { predict } from './api/client'
import type { PredictionResponse } from './api/types'
import { CategoryBars } from './components/CategoryBars'

const ALL_CATEGORIES = [
  'Arts, Crafts & Sewing',
  'Cell Phones & Accessories',
  'Clothing, Shoes & Jewelry',
  'Tools & Home Improvement',
  'Health & Personal Care',
  'Baby Products',
  'Baby',
  'Patio, Lawn & Garden',
  'Beauty',
  'Sports & Outdoors',
  'Electronics',
  'All Electronics',
  'Automotive',
  'Toys & Games',
  'All Beauty',
  'Office Products',
  'Appliances',
  'Musical Instruments',
  'Industrial & Scientific',
  'Grocery & Gourmet Food',
  'Pet Supplies',
]

export default function App() {
  const [title, setTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [emptyHint, setEmptyHint] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)

    const trimmed = title.trim()
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
            Введите title товара или объявления <span className="font-medium text-zinc-800">на английском языке</span>
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
          <label htmlFor="title" className="block text-sm font-medium text-zinc-800">
            Title{' '}
            <span className="font-normal text-amber-800">(только английский текст)</span>
          </label>
          <textarea
            id="title"
            name="title"
            rows={6}
            value={title}
            onChange={(event) => {
              setTitle(event.target.value)
              if (emptyHint && event.target.value.trim()) {
                setEmptyHint(false)
              }
            }}
            placeholder="e.g. Noise cancelling over-ear headphones with charging case"
            className="mt-2 w-full resize-y rounded-xl border border-zinc-300 bg-white px-3 py-2 text-sm text-zinc-900 shadow-inner outline-none ring-indigo-500 transition focus:border-indigo-500 focus:ring-2"
            aria-invalid={emptyHint}
            aria-describedby={emptyHint ? 'title-empty' : undefined}
          />
          {emptyHint ? (
            <p id="title-empty" className="mt-2 text-sm text-red-600" role="alert">
              Введите непустой title, чтобы получить предсказание.
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

        <section className="mt-10 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
          <h3 className="text-base font-semibold text-zinc-900">Все возможные категории</h3>
          <p className="mt-1 text-sm text-zinc-600">
            Категории, на которые обучена текущая модель.
          </p>
          <ul className="mt-4 grid gap-2 sm:grid-cols-2">
            {ALL_CATEGORIES.map((category) => (
              <li
                key={category}
                className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-800"
              >
                {category}
              </li>
            ))}
          </ul>
        </section>
      </main>

      <footer className="border-t border-zinc-200 bg-white py-8 text-center text-xs text-zinc-500">
        Учебный full-stack demo: React + FastAPI + Docker
      </footer>
    </div>
  )
}
