import type { BatchResponse, CustomerPayload, HealthResponse, ScoreResult } from '../types/api'
import { resolveApiBaseUrl } from './config'

export const API_BASE_URL = resolveApiBaseUrl(import.meta.env.VITE_API_URL)
const REQUEST_BASE_URL = import.meta.env.DEV && API_BASE_URL === 'http://localhost:8000' ? '/api' : API_BASE_URL
const REQUEST_TIMEOUT_MS = 15_000

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    const response = await fetch(`${REQUEST_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
    const body: unknown = await response.json().catch(() => null)
    if (!response.ok) {
      const detail = typeof body === 'object' && body !== null && 'detail' in body
        ? (body as { detail: unknown }).detail
        : undefined
      const message = Array.isArray(detail)
        ? detail.map((item) => typeof item === 'object' && item !== null && 'msg' in item ? String(item.msg) : String(item)).join('; ')
        : typeof detail === 'string' ? detail : `Request failed (${response.status})`
      throw new ApiError(message, response.status)
    }
    return body as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') throw new ApiError('The API request timed out. Check the service and try again.')
    if (error instanceof TypeError) throw new ApiError('Cannot reach the ChurnGuard API. Check that it is running and the API URL is correct.')
    throw new ApiError('The API returned an unreadable response.')
  } finally {
    window.clearTimeout(timer)
  }
}

function isScoreResult(value: unknown): value is ScoreResult {
  if (typeof value !== 'object' || value === null) return false
  const row = value as Record<string, unknown>
  return ['churn_probability', 'predicted_90d_value', 'risk_weighted_value'].every((key) => typeof row[key] === 'number' && Number.isFinite(row[key] as number))
}

export const api = {
  async health(): Promise<HealthResponse> {
    const result = await request<HealthResponse>('/health')
    if (typeof result !== 'object' || result === null || typeof result.status !== 'string' || typeof result.churn_model_loaded !== 'boolean'
      || typeof result.calibrator_loaded !== 'boolean' || typeof result.future_value_model_loaded !== 'boolean') {
      throw new ApiError('The API health response did not match the expected format.')
    }
    return result
  },
  async predictCustomer(payload: CustomerPayload): Promise<ScoreResult> {
    const result: unknown = await request('/predict', { method: 'POST', body: JSON.stringify(payload) })
    if (!isScoreResult(result)) throw new ApiError('The API prediction response did not match the expected format.')
    return result
  },
  async batchScore(customers: CustomerPayload[]): Promise<BatchResponse> {
    const result: unknown = await request('/batch_score', { method: 'POST', body: JSON.stringify({ customers }) })
    if (typeof result !== 'object' || result === null || !('scored_customers' in result)
      || !Array.isArray((result as BatchResponse).scored_customers)
      || typeof (result as BatchResponse).count !== 'number'
      || (result as BatchResponse).count !== (result as BatchResponse).scored_customers.length
      || !(result as BatchResponse).scored_customers.every(isScoreResult)) {
      throw new ApiError('The API batch response did not match the expected format.')
    }
    return result as BatchResponse
  },
}
