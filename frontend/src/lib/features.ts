import type { CustomerPayload, FeatureValue } from '../types/api'

export const FEATURE_GROUPS = [
  { title: 'Purchase behaviour', fields: ['recency_days', 'purchase_invoice_count', 'purchase_line_count', 'units_purchased', 'historical_net_spend', 'tenure_days', 'average_order_value', 'average_items_per_order', 'distinct_products'] },
  { title: 'Purchase timing', fields: ['mean_inter_purchase_gap_days', 'std_inter_purchase_gap_days', 'inactive_days'] },
  { title: 'Recent activity', fields: ['recent_3m_net_spend', 'previous_3m_net_spend', 'recent_3m_order_count', 'previous_3m_order_count', 'activity_3m_order_count', 'activity_6m_order_count', 'active_3m', 'active_6m'] },
  { title: 'Trends', fields: ['spend_trend_difference_3m', 'spend_trend_ratio_3m', 'order_trend_difference_3m', 'order_trend_ratio_3m'] },
  { title: 'Returns & calendar', fields: ['return_quantity', 'return_value', 'negative_adjustment_rate', 'snapshot_month'] },
] as const

export const FEATURE_COLUMNS = [
  'recency_days', 'purchase_invoice_count', 'purchase_line_count', 'units_purchased', 'historical_net_spend',
  'log1p_purchase_invoice_count', 'signed_log1p_historical_net_spend', 'tenure_days', 'average_order_value',
  'average_items_per_order', 'distinct_products', 'mean_inter_purchase_gap_days', 'std_inter_purchase_gap_days',
  'recent_3m_net_spend', 'previous_3m_net_spend', 'spend_trend_difference_3m', 'spend_trend_ratio_3m',
  'recent_3m_order_count', 'previous_3m_order_count', 'order_trend_difference_3m', 'order_trend_ratio_3m',
  'return_quantity', 'return_value', 'negative_adjustment_rate', 'snapshot_month', 'inactive_days', 'country',
  'activity_3m_order_count', 'active_3m', 'activity_6m_order_count', 'active_6m',
] as const

// Options are copied from the current, ignored models/metadata.json artifact.
// Keep these aligned when inference artifacts are refreshed: the API remains authoritative.
export const COUNTRIES = ['United Kingdom', 'France', 'Australia', 'Netherlands', 'Germany', 'Norway', 'EIRE', 'Switzerland', 'Spain', 'Poland', 'Portugal', 'Italy', 'Belgium', 'Lithuania', 'Japan', 'Iceland', 'Channel Islands', 'Denmark', 'Austria', 'Sweden', 'Finland', 'Cyprus', 'Greece', 'Singapore', 'Lebanon', 'United Arab Emirates', 'Israel', 'Saudi Arabia', 'Czech Republic', 'Canada'] as const

export const DEMO_PROFILE: Record<string, FeatureValue> = {
  recency_days: 24, purchase_invoice_count: 8, purchase_line_count: 34, units_purchased: 420,
  historical_net_spend: 1240.5, log1p_purchase_invoice_count: Math.log1p(8),
  signed_log1p_historical_net_spend: Math.log1p(1240.5), tenure_days: 340, average_order_value: 155.06,
  average_items_per_order: 52.5, distinct_products: 29, mean_inter_purchase_gap_days: 41.2,
  std_inter_purchase_gap_days: 18.5, recent_3m_net_spend: 280.4, previous_3m_net_spend: 510.2,
  spend_trend_difference_3m: -229.8, spend_trend_ratio_3m: 0.54998, recent_3m_order_count: 2,
  previous_3m_order_count: 4, order_trend_difference_3m: -2, order_trend_ratio_3m: 0.5,
  return_quantity: 6, return_value: 18.2, negative_adjustment_rate: 0.0147, snapshot_month: 6,
  inactive_days: 24, country: 'United Kingdom', activity_3m_order_count: 2, active_3m: 1,
  activity_6m_order_count: 6, active_6m: 1,
}

export function validateFeatureRecord(features: Record<string, unknown>): string[] {
  const errors: string[] = []
  const missing = FEATURE_COLUMNS.filter((column) => !(column in features))
  const extra = Object.keys(features).filter((column) => !FEATURE_COLUMNS.includes(column as typeof FEATURE_COLUMNS[number]))
  if (missing.length) errors.push(`Missing required features: ${missing.join(', ')}`)
  if (extra.length) errors.push(`Unexpected features: ${extra.join(', ')}`)
  for (const key of FEATURE_COLUMNS) {
    const value = features[key]
    if (typeof value === 'number' && !Number.isFinite(value)) errors.push(`${key} must be a finite number`)
    else if (value === null || value === undefined || value === '') errors.push(`${key} is required`)
  }
  if (typeof features.country === 'string' && !COUNTRIES.includes(features.country as typeof COUNTRIES[number])) errors.push('country must match a category supported by the current model')
  return errors
}

export interface CsvValidation {
  errors: string[]
  warnings: string[]
  headers: string[]
  records: CustomerPayload[]
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = []
  let value = ''; let quoted = false
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i]
    if (char === '"' && quoted && line[i + 1] === '"') { value += '"'; i += 1 }
    else if (char === '"') quoted = !quoted
    else if (char === ',' && !quoted) { cells.push(value.trim()); value = '' }
    else value += char
  }
  if (quoted) throw new Error('CSV contains an unclosed quoted field.')
  cells.push(value.trim())
  return cells
}

function isIsoCalendarDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const parsed = new Date(`${value}T00:00:00Z`)
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value
}

export function validateCsv(text: string): CsvValidation {
  const errors: string[] = []; const warnings: string[] = []
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length < 2) return { errors: ['CSV must include a header and at least one customer row.'], warnings, headers: [], records: [] }
  let headers: string[]
  let rows: string[][]
  try { headers = parseCsvLine(lines[0]); rows = lines.slice(1).map(parseCsvLine) }
  catch (error) { return { errors: [error instanceof Error ? error.message : 'CSV could not be parsed.'], warnings, headers: [], records: [] } }
  if (new Set(headers).size !== headers.length) errors.push('CSV contains duplicate column names.')
  const required = FEATURE_COLUMNS.filter((column) => !headers.includes(column))
  if (required.length) errors.push(`${required.length} required columns are missing: ${required.join(', ')}`)
  const optional = ['CustomerID', 'snapshot_date']
  const unknown = headers.filter((header) => !FEATURE_COLUMNS.includes(header as typeof FEATURE_COLUMNS[number]) && !optional.includes(header))
  if (unknown.length) errors.push(`Unexpected columns: ${unknown.join(', ')}`)
  if (rows.some((row) => row.length !== headers.length)) errors.push('Every CSV row must have the same number of values as the header.')
  const records: CustomerPayload[] = []
  rows.forEach((row, index) => {
    if (row.length !== headers.length) return
    const raw = Object.fromEntries(headers.map((key, i) => [key, row[i]]))
    const features: Record<string, unknown> = {}
    for (const name of FEATURE_COLUMNS) {
      if (name === 'country') features[name] = raw[name]
      else {
        const value = raw[name]
        const number = Number(value)
        if (value === '' || !Number.isFinite(number)) errors.push(`Row ${index + 2}: ${name} must be a finite number.`)
        features[name] = number
      }
    }
    if (typeof features.country === 'string' && !COUNTRIES.includes(features.country as typeof COUNTRIES[number])) errors.push(`Row ${index + 2}: country is not supported by the current model.`)
    if (raw.snapshot_date && !isIsoCalendarDate(raw.snapshot_date)) errors.push(`Row ${index + 2}: snapshot_date must be a valid date in YYYY-MM-DD format.`)
    const customer = raw.CustomerID
    if (customer && !Number.isFinite(Number(customer)) && customer.length > 100) errors.push(`Row ${index + 2}: CustomerID is too long.`)
    const record: CustomerPayload = { features: features as Record<string, FeatureValue> }
    if (customer) record.CustomerID = Number.isFinite(Number(customer)) ? Number(customer) : customer
    if (raw.snapshot_date) record.snapshot_date = raw.snapshot_date
    const rowErrors = validateFeatureRecord(features)
    if (rowErrors.length) errors.push(...rowErrors.map((message) => `Row ${index + 2}: ${message}`))
    records.push(record)
  })
  if (!errors.length) warnings.push(`Ready to score ${records.length.toLocaleString()} customer${records.length === 1 ? '' : 's'}.`)
  return { errors: [...new Set(errors)], warnings, headers, records: errors.length ? [] : records }
}

export function formatCurrency(value: number, currency = 'GBP'): string {
  return new Intl.NumberFormat('en-GB', { style: 'currency', currency, maximumFractionDigits: 0 }).format(value)
}

export function formatProbability(value: number): string {
  return new Intl.NumberFormat('en-GB', { style: 'percent', maximumFractionDigits: 1 }).format(value)
}

export function snapshotMonthFromDate(value: string): number | null {
  if (!isIsoCalendarDate(value)) return null
  return new Date(`${value}T00:00:00Z`).getUTCMonth() + 1
}

export function quoteCsv(value: unknown): string {
  const text = String(value ?? '')
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}
