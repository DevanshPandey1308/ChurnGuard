import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ScoreResult } from '../types/api'
import { Analytics } from './Analytics'

describe('batch-scoped analytics', () => {
  it('summarizes and lists only the scored customer results in query state', () => {
    const client = new QueryClient()
    const rows: ScoreResult[] = [
      { CustomerID: 101, snapshot_date: '2011-06-01', churn_probability: 0.8, predicted_90d_value: 100, risk_weighted_value: 80, churn_rank: 1, value_rank: 2, risk_value_rank: 1 },
      { CustomerID: 102, snapshot_date: '2011-06-01', churn_probability: 0.3, predicted_90d_value: 250, risk_weighted_value: 75, churn_rank: 2, value_rank: 1, risk_value_rank: 2 },
    ]
    client.setQueryData(['batch-results'], rows)
    render(<QueryClientProvider client={client}><Analytics /></QueryClientProvider>)
    expect(screen.getByText('2 customers')).toBeInTheDocument()
    expect(screen.getByText('Scored customer snapshots')).toBeInTheDocument()
    expect(screen.getByText('CustomerID')).toBeInTheDocument()
    expect(screen.getAllByText('80%')).toHaveLength(2)
    expect(screen.getAllByText('£250')).toHaveLength(2)
    expect(screen.getByRole('cell', { name: '101' })).toBeInTheDocument()
  })
})

