import type { PredictionResponse } from './types'

function getBaseUrl(): string {
  const raw = import.meta.env.VITE_API_BASE_URL
  if (typeof raw === 'string' && raw.length > 0) {
    return raw.replace(/\/$/, '')
  }
  return 'http://localhost:8000'
}

export async function predict(description: string): Promise<PredictionResponse> {
  const base = getBaseUrl()
  const response = await fetch(`${base}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ description }),
  })

  const text = await response.text()

  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    if (text) {
      try {
        const parsed = JSON.parse(text) as { detail?: unknown }
        if (parsed.detail !== undefined) {
          detail =
            typeof parsed.detail === 'string'
              ? parsed.detail
              : JSON.stringify(parsed.detail)
        } else {
          detail = text
        }
      } catch {
        detail = text
      }
    }
    throw new Error(detail)
  }

  return JSON.parse(text) as PredictionResponse
}
