import { describe, expect, it } from 'vitest'
import { COUNTRIES, DEMO_PROFILE, FEATURE_COLUMNS, formatCurrency, formatProbability, snapshotMonthFromDate, validateCsv, validateFeatureRecord } from './features'
import type { FeatureValue } from '../types/api'
import { resolveApiBaseUrl } from '../api/config'

describe('frontend configuration and feature utilities', () => {
  it('uses local API default and normalizes configured URL', () => {
    expect(resolveApiBaseUrl()).toBe('http://localhost:8000')
    expect(resolveApiBaseUrl(' https://api.example.test/ ')).toBe('https://api.example.test')
  })

  it('validates the exact required inference feature set and supported country', () => {
    const profile: Record<string, FeatureValue> = { ...DEMO_PROFILE, log1p_purchase_invoice_count: Math.log1p(Number(DEMO_PROFILE.purchase_invoice_count)), signed_log1p_historical_net_spend: Math.log1p(Number(DEMO_PROFILE.historical_net_spend)) }
    expect(FEATURE_COLUMNS).toHaveLength(31)
    expect(COUNTRIES).toContain(profile.country)
    expect(validateFeatureRecord(profile)).toEqual([])
    expect(validateFeatureRecord({ ...profile, future_net_spend_90d: 10 })).toContain('Unexpected features: future_net_spend_90d')
  })

  it('validates missing columns and numeric/categorical CSV values before API submission', () => {
    const missing = validateCsv('CustomerID,snapshot_date\n123,2011-06-01')
    expect(missing.errors[0]).toContain('required columns are missing')
    const header = [...FEATURE_COLUMNS, 'CustomerID', 'snapshot_date'].join(',')
    const values = FEATURE_COLUMNS.map((key) => key === 'country' ? 'United Kingdom' : String(DEMO_PROFILE[key as keyof typeof DEMO_PROFILE] ?? (key === 'log1p_purchase_invoice_count' ? Math.log1p(8) : Math.log1p(1240.5))))
    const valid = validateCsv(`${header}\n${[...values, '123', '2011-06-01'].join(',')}`)
    expect(valid.errors).toEqual([])
    expect(valid.records).toHaveLength(1)
    expect(valid.records[0].CustomerID).toBe(123)
    const invalid = validateCsv(`${header}\n${[...values.slice(0, FEATURE_COLUMNS.indexOf('country')), 'Made Up', ...values.slice(FEATURE_COLUMNS.indexOf('country') + 1), '123', '2011-06-01'].join(',')}`)
    expect(invalid.errors.some((message) => message.includes('country is not supported'))).toBe(true)
    const invalidDate = validateCsv(`${header}\n${[...values, '123', '2011-02-30'].join(',')}`)
    expect(invalidDate.errors.some((message) => message.includes('snapshot_date must be a valid date'))).toBe(true)
  })

  it('formats values without unnecessary precision', () => {
    expect(formatCurrency(780.09)).toBe('£780')
    expect(formatProbability(0.735)).toBe('73.5%')
    expect(snapshotMonthFromDate('2011-06-01')).toBe(6)
    expect(snapshotMonthFromDate('2011-02-30')).toBeNull()
  })
})
