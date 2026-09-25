import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './client'

afterEach(() => vi.unstubAllGlobals())

describe('typed ChurnGuard API client', () => {
  it('sends the exact single-customer request to /predict and parses scores', async () => {
    const payload = { CustomerID: 17, snapshot_date: '2011-06-01', features: { recency_days: 12 } }
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...payload, snapshot_date: '2011-06-01T00:00:00', churn_probability: 0.7, predicted_90d_value: 120, risk_weighted_value: 84 }), { status: 200 }))
    vi.stubGlobal('fetch', fetcher)
    const result = await api.predictCustomer(payload)
    expect(fetcher).toHaveBeenCalledWith('/api/predict', expect.objectContaining({ method: 'POST', body: JSON.stringify(payload) }))
    expect(result.churn_probability).toBe(0.7)
    expect(result.predicted_90d_value).toBe(120)
  })

  it('sends batch records in the backend customers envelope', async () => {
    const records = [{ CustomerID: 5, features: { recency_days: 1 } }]
    const resultRow = { CustomerID: 5, snapshot_date: null, churn_probability: 0.5, predicted_90d_value: 10, risk_weighted_value: 5, churn_rank: 1 }
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ count: 1, scored_customers: [resultRow] }), { status: 200 }))
    vi.stubGlobal('fetch', fetcher)
    const response = await api.batchScore(records)
    expect(fetcher).toHaveBeenCalledWith('/api/batch_score', expect.objectContaining({ body: JSON.stringify({ customers: records }) }))
    expect(response.scored_customers[0].churn_rank).toBe(1)
  })

  it('parses FastAPI detail errors and rejects malformed score responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Invalid feature' }), { status: 422 })))
    await expect(api.health()).rejects.toMatchObject({ status: 422, message: 'Invalid feature' })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ churn_probability: 0.2 }), { status: 200 })))
    await expect(api.predictCustomer({ features: {} })).rejects.toThrow('did not match the expected format')
  })

  it('validates health response shape', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'ok', churn_model_loaded: true, calibrator_loaded: true, future_value_model_loaded: true }), { status: 200 })))
    expect((await api.health()).future_value_model_loaded).toBe(true)
  })
})
