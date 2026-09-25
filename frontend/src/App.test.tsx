import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

describe('application navigation and overview', () => {
  it('renders the product shell and overview without a fabricated KPI', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'ok', churn_model_loaded: true, calibrator_loaded: true, future_value_model_loaded: true }), { status: 200 })))
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}><App /></MemoryRouter></QueryClientProvider>)
    expect(screen.getAllByText('ChurnGuard').length).toBeGreaterThan(0)
    expect(await screen.findByText('Operational')).toBeInTheDocument()
    expect(screen.getByText('Risk-Weighted Value')).toBeInTheDocument()
    expect(screen.getByText(/not guaranteed recoverable revenue, ROI/i)).toBeInTheDocument()
  })
})
