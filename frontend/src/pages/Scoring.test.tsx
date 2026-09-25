import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Scoring } from './Scoring'

afterEach(() => vi.unstubAllGlobals())

describe('single-customer inference result', () => {
  it('submits the current feature snapshot and presents formatted API results', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      CustomerID: '10001', snapshot_date: '2011-06-01T00:00:00', churn_probability: 0.735,
      predicted_90d_value: 780.09, risk_weighted_value: 573.36,
    }), { status: 200 }))
    vi.stubGlobal('fetch', fetcher)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const user = userEvent.setup()
    render(<QueryClientProvider client={client}><MemoryRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><Scoring /></MemoryRouter></QueryClientProvider>)
    await user.click(screen.getByRole('button', { name: 'Score customer' }))
    expect(await screen.findByText('73.5%')).toBeInTheDocument()
    expect(screen.getByText('£780')).toBeInTheDocument()
    expect(screen.getByText('£573')).toBeInTheDocument()
    expect(screen.getByText(/Customer #10001/)).toBeInTheDocument()
    const sent = JSON.parse(String(fetcher.mock.calls[0][1]?.body)) as { features: Record<string, unknown> }
    expect(Object.keys(sent.features)).toHaveLength(31)
    expect(sent.features.snapshot_month).toBe(6)
    expect(sent.features).not.toHaveProperty('future_net_spend_90d')
  })
})
