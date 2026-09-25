import { useMemo, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Activity, ArrowDownToLine, ArrowUpDown, Check, ChevronDown, Search, UploadCloud } from 'lucide-react'
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { api, ApiError } from '../api/client'
import { Button, FieldLabel, PageHeader, Panel, StateMessage } from '../components/ui'
import type { ScoreResult } from '../types/api'
import { COUNTRIES, DEMO_PROFILE, FEATURE_COLUMNS, FEATURE_GROUPS, formatCurrency, formatProbability, quoteCsv, snapshotMonthFromDate, validateCsv, validateFeatureRecord } from '../lib/features'

type Mode = 'single' | 'batch'
const LABELS: Record<string, string> = {
  recency_days: 'Recency (days)', purchase_invoice_count: 'Purchase invoices', purchase_line_count: 'Purchase lines', units_purchased: 'Units purchased', historical_net_spend: 'Historical net spend', tenure_days: 'Tenure (days)', average_order_value: 'Average order value', average_items_per_order: 'Average items per order', distinct_products: 'Distinct products', mean_inter_purchase_gap_days: 'Mean purchase gap (days)', std_inter_purchase_gap_days: 'Purchase gap deviation (days)', inactive_days: 'Inactive days', recent_3m_net_spend: 'Recent 3-month net spend', previous_3m_net_spend: 'Previous 3-month net spend', recent_3m_order_count: 'Recent 3-month orders', previous_3m_order_count: 'Previous 3-month orders', activity_3m_order_count: 'Activity: 3-month orders', activity_6m_order_count: 'Activity: 6-month orders', active_3m: 'Active in 3 months (0/1)', active_6m: 'Active in 6 months (0/1)', spend_trend_difference_3m: 'Spend trend difference', spend_trend_ratio_3m: 'Spend trend ratio', order_trend_difference_3m: 'Order trend difference', order_trend_ratio_3m: 'Order trend ratio', return_quantity: 'Return quantity', return_value: 'Return value', negative_adjustment_rate: 'Negative adjustment rate', snapshot_month: 'Snapshot month',
}

export function Scoring() {
  const [mode, setMode] = useState<Mode>('single')
  return <>
    <PageHeader eyebrow="CUSTOMER SCORING" title="Score customers" description="Submit the latest point-in-time feature snapshot to the ChurnGuard inference API." />
    <div className="segmented" role="tablist" aria-label="Scoring mode"><button role="tab" aria-selected={mode === 'single'} onClick={() => setMode('single')}>Single customer</button><button role="tab" aria-selected={mode === 'batch'} onClick={() => setMode('batch')}>Batch scoring</button></div>
    {mode === 'single' ? <SingleCustomer /> : <BatchScoring />}
  </>
}

function SingleCustomer() {
  const [customerId, setCustomerId] = useState('10001')
  const [snapshotDate, setSnapshotDate] = useState('2011-06-01')
  const [form, setForm] = useState<Record<string, string>>(() => Object.fromEntries(Object.entries(DEMO_PROFILE).map(([key, value]) => [key, String(value)])))
  const [submitted, setSubmitted] = useState(false)
  const mutation = useMutation({ mutationFn: api.predictCustomer })
  const features = useMemo(() => {
    const values: Record<string, number | string> = {}
    for (const column of FEATURE_COLUMNS) {
      if (column === 'log1p_purchase_invoice_count') values[column] = Math.log1p(Number(form.purchase_invoice_count))
      else if (column === 'signed_log1p_historical_net_spend') { const amount = Number(form.historical_net_spend); values[column] = Math.sign(amount) * Math.log1p(Math.abs(amount)) }
      else if (column === 'snapshot_month') values[column] = snapshotMonthFromDate(snapshotDate) ?? Number.NaN
      else if (column === 'country') values[column] = form.country
      else values[column] = form[column]?.trim() === '' ? Number.NaN : Number(form[column])
    }
    return values
  }, [form, snapshotDate])
  const errors = submitted ? validateFeatureRecord(features) : []
  const setField = (key: string, value: string) => setForm((old) => ({ ...old, [key]: value }))
  const submit = () => {
    setSubmitted(true)
    const featureErrors = validateFeatureRecord(features)
    if (featureErrors.length || !snapshotDate || Number.isNaN(Date.parse(snapshotDate))) return
    mutation.mutate({ CustomerID: customerId || undefined, snapshot_date: snapshotDate, features })
  }
  return <div className="scoring-layout">
    <Panel className="form-panel"><div className="panel-heading"><div><h2>Customer snapshot</h2><p>Enter the customer's latest point-in-time feature snapshot.</p></div><div className="example-actions"><span className="demo-badge">Demonstration values</span><Button className="secondary" type="button" onClick={() => { setForm(Object.fromEntries(Object.entries(DEMO_PROFILE).map(([key, value]) => [key, String(value)]))); setCustomerId('10001'); setSnapshotDate('2011-06-01'); setSubmitted(false); mutation.reset() }}>Load example profile</Button></div></div>
      <div className="form-intro"><span>Customer behaviour observed up to the selected snapshot date.</span><span>This feature-level API does not accept raw transaction records.</span></div>
      <div className="context-grid"><label><FieldLabel name="CustomerID" label="Customer ID" /><input value={customerId} onChange={(event) => setCustomerId(event.target.value)} /></label><label><FieldLabel name="snapshot_date" label="Snapshot date" hint="Feature values are measured as of this date." /><input type="date" value={snapshotDate} onChange={(event) => setSnapshotDate(event.target.value)} /></label></div>
      <div className="country-select"><label><FieldLabel name="country" label="Country" hint="Options from the current inference artifact schema." /><select value={form.country} onChange={(event) => setField('country', event.target.value)}>{COUNTRIES.map((country) => <option key={country}>{country}</option>)}</select></label></div>
      {FEATURE_GROUPS.map((group, index) => <details className="feature-group" key={group.title} open={index === 0}><summary>{group.title}<ChevronDown size={16} /></summary><div className="field-grid">{group.fields.map((name) => <label key={name}><FieldLabel name={name} label={LABELS[name]} />{name === 'active_3m' || name === 'active_6m' ? <select value={form[name]} onChange={(event) => setField(name, event.target.value)}><option value="1">Active</option><option value="0">Inactive</option></select> : <input type="number" step="any" value={name === 'snapshot_month' ? snapshotMonthFromDate(snapshotDate) ?? '' : form[name] ?? ''} disabled={name === 'snapshot_month'} onChange={(event) => setField(name, event.target.value)} />}</label>)}</div></details>)}
      <div className="form-footer"><span className="muted">Derived log features are calculated to match the current backend contract.</span><Button className="primary" type="button" disabled={mutation.isPending} onClick={submit}>{mutation.isPending ? 'Scoring customer…' : 'Score customer'}</Button></div>
      {submitted && (!snapshotDate || Number.isNaN(Date.parse(snapshotDate))) && <StateMessage kind="error">Enter a valid snapshot date.</StateMessage>}
      {errors.length > 0 && <StateMessage kind="error">{errors[0]}{errors.length > 1 ? ` (and ${errors.length - 1} more)` : ''}</StateMessage>}
      {mutation.isError && <StateMessage kind="error">{friendlyError(mutation.error)}</StateMessage>}
    </Panel>
    <div className="result-column">{mutation.isPending ? <Panel><StateMessage kind="loading">Scoring customer…</StateMessage></Panel> : mutation.data ? <PredictionResult result={mutation.data} /> : <Panel className="empty-result"><div className="empty-icon"><Activity size={19} /></div><h3>Your prediction will appear here</h3><p>Complete the feature snapshot and send it to the inference API to see the returned churn, value, and prioritization estimates.</p><p className="fineprint">Example profile is demo input only. It does not represent model performance evidence.</p></Panel>}</div>
  </div>
}

function PredictionResult({ result }: { result: ScoreResult }) {
  const snapshotLabel = result.snapshot_date ? new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).format(new Date(result.snapshot_date)) : 'Not supplied'
  const probabilityWidth = `${Math.min(1, Math.max(0, result.churn_probability)) * 100}%`
  return <Panel className="result-panel"><div className="eyebrow">PREDICTION · API RESPONSE</div><h2>Customer #{result.CustomerID ?? '—'}</h2><div className="result-snapshot">Snapshot · {snapshotLabel}</div>
    <div className="result-metrics">
      <section className="result-metric primary-metric"><div className="metric-label">Churn Probability</div><strong>{formatProbability(result.churn_probability)}</strong><small>Estimated probability of non-repurchase during the next 90 days.</small><div className="probability-meter" role="meter" aria-label="Churn probability indicator" aria-valuemin={0} aria-valuemax={1} aria-valuenow={result.churn_probability}><span style={{ width: probabilityWidth }} /></div></section>
      <section className="result-metric"><div className="metric-label">Predicted 90-Day Value</div><strong>{formatCurrency(result.predicted_90d_value)}</strong><small>Model-predicted future net spend.</small></section>
      <section className="result-metric"><div className="metric-label">Risk-Weighted Value</div><strong>{formatCurrency(result.risk_weighted_value)}</strong><small>Prioritization signal combining churn probability and predicted future value.</small></section>
    </div><div className="result-interpretation"><div className="interpretation-label">INTERPRETATION</div><p>The model estimates non-repurchase probability at {formatProbability(result.churn_probability)}, with {formatCurrency(result.predicted_90d_value)} predicted 90-day net spend. The returned risk-weighted value is {formatCurrency(result.risk_weighted_value)}.</p></div><p className="fineprint">Risk-weighted value is a prioritization metric. It is not guaranteed recoverable revenue, ROI, or a measured intervention effect.</p></Panel>
}

function BatchScoring() {
  const [fileName, setFileName] = useState('')
  const [parsing, setParsing] = useState(false)
  const [validation, setValidation] = useState<ReturnType<typeof validateCsv> | null>(null)
  const [dragging, setDragging] = useState(false)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const [sort, setSort] = useState<keyof ScoreResult>('risk_value_rank')
  const [ascending, setAscending] = useState(true)
  const queryClient = useQueryClient()
  const mutation = useMutation({ mutationFn: api.batchScore, onSuccess: (response) => queryClient.setQueryData(['batch-results'], response.scored_customers) })
  const parseFile = async (file?: File) => {
    if (!file) return
    setFileName(file.name); setValidation(null); mutation.reset()
    if (!file.name.toLowerCase().endsWith('.csv')) { setValidation({ errors: ['Choose a .csv file.'], warnings: [], headers: [], records: [] }); return }
    if (file.size > 10 * 1024 * 1024) { setValidation({ errors: ['CSV must be smaller than 10 MB.'], warnings: [], headers: [], records: [] }); return }
    setParsing(true)
    try { const text = await file.text(); setValidation(validateCsv(text)) }
    catch { setValidation({ errors: ['The selected file could not be read. Try another CSV file.'], warnings: [], headers: [], records: [] }) }
    finally { setParsing(false) }
  }
  const filtered = useMemo(() => {
    const rows = (mutation.data?.scored_customers ?? []).filter((row) => `${row.CustomerID ?? ''} ${row.snapshot_date ?? ''}`.toLowerCase().includes(search.toLowerCase()))
    return rows.sort((a, b) => { const av = a[sort]; const bv = b[sort]; const comparison = typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av ?? '').localeCompare(String(bv ?? '')); return ascending ? comparison : -comparison })
  }, [mutation.data, search, sort, ascending])
  const current = filtered.slice(page * 10, page * 10 + 10)
  const results = mutation.data?.scored_customers ?? []
  const score = () => { if (!validation || validation.errors.length) return; mutation.mutate(validation.records) }
  const downloadTemplate = () => downloadCsv(FEATURE_COLUMNS.join(',') + ',CustomerID,snapshot_date\n')
  const exportResults = () => {
    const headers = ['CustomerID', 'snapshot_date', 'churn_probability', 'predicted_90d_value', 'risk_weighted_value', 'churn_rank', 'value_rank', 'risk_value_rank']
    downloadCsv([headers.join(','), ...results.map((row) => headers.map((key) => quoteCsv(row[key as keyof ScoreResult])).join(','))].join('\n'), 'churnguard-scored-customers.csv')
  }
  return <div className="batch-layout">
    <Panel className="batch-upload"><div className="panel-heading"><div><h2>Upload customer features</h2><p>CSV with the exact 31 inference feature columns. CustomerID and snapshot_date are optional metadata.</p></div><Button className="secondary" onClick={downloadTemplate}><ArrowDownToLine size={15} />Download CSV template</Button></div>
      <div className="batch-steps" aria-label="Batch scoring workflow">{['Upload', 'Validate', 'Preview', 'Score', 'Results', 'Export'].map((label, index) => { const done = index === 0 ? Boolean(fileName) : index === 1 ? Boolean(validation && validation.errors.length === 0) : index === 2 ? Boolean(validation && validation.records.length > 0) : index >= 4 ? Boolean(mutation.data) : Boolean(mutation.data); const error = index === 1 && Boolean(validation?.errors.length); const current = !done && !error && (index === 0 || (index === 1 && fileName && !parsing) || (index === 3 && validation && validation.errors.length === 0) || (index === 4 && mutation.isPending)); return <div className={`batch-step ${done ? 'done' : error ? 'error' : current ? 'current' : ''}`} key={label}><span>{done ? '✓' : error ? '!' : `0${index + 1}`}</span>{label}</div> })}</div>
      <label className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); void parseFile(event.dataTransfer.files[0]) }}>
        <UploadCloud size={22} /><strong>{fileName || 'Drop a CSV file here'}</strong><span>or browse from your device · maximum 10 MB</span><input type="file" accept=".csv,text/csv" aria-label="Choose CSV file" onChange={(event) => void parseFile(event.target.files?.[0])} />
      </label>
      {parsing && <StateMessage kind="loading">Reading and validating CSV…</StateMessage>}
      {validation && <div className={`validation-box ${validation.errors.length ? 'invalid' : 'valid'}`}>{validation.errors.length ? <><b>File needs attention</b><ul>{validation.errors.slice(0, 8).map((error) => <li key={error}>{error}</li>)}</ul>{validation.errors.length > 8 && <small>And {validation.errors.length - 8} more issues.</small>}</> : <><b><Check size={15} /> Ready to score</b><p>{validation.records.length.toLocaleString()} customers passed validation.</p><p className="muted">{validation.headers.length} columns · first rows shown in preview below</p></>}</div>}
      {validation && validation.records.length > 0 && <div className="csv-preview"><div className="eyebrow">INPUT PREVIEW · FIRST {Math.min(5, validation.records.length)} ROWS</div><div className="table-wrap"><table><thead><tr><th>Customer ID</th><th>Snapshot date</th><th>Country</th><th>Recency</th><th>Historical net spend</th></tr></thead><tbody>{validation.records.slice(0, 5).map((record, index) => <tr key={`${record.CustomerID ?? 'customer'}-${index}`}><td>{record.CustomerID ?? '—'}</td><td>{record.snapshot_date ?? '—'}</td><td>{record.features.country}</td><td>{record.features.recency_days}</td><td>{formatCurrency(Number(record.features.historical_net_spend))}</td></tr>)}</tbody></table></div></div>}
      {mutation.isError && <StateMessage kind="error">{friendlyError(mutation.error)}</StateMessage>}
      <div className="batch-actions"><Button className="primary" disabled={!validation || validation.errors.length > 0 || mutation.isPending} onClick={score}>{mutation.isPending ? `Scoring ${validation?.records.length.toLocaleString()} customers…` : 'Score customers'}</Button><span className="muted">Inputs are validated in this browser; the API checks them again.</span></div>
    </Panel>
    {mutation.isPending && <Panel><StateMessage kind="loading">Scoring {validation?.records.length.toLocaleString()} customers…</StateMessage></Panel>}
    {results.length > 0 && <>
      <div className="batch-kpis"><Summary label="Customers scored" value={results.length.toLocaleString()} /><Summary label="Highest churn probability" value={formatProbability(Math.max(...results.map((row) => row.churn_probability)))} /><Summary label="Highest predicted value" value={formatCurrency(Math.max(...results.map((row) => row.predicted_90d_value)))} /><Summary label="Highest risk-weighted value" value={formatCurrency(Math.max(...results.map((row) => row.risk_weighted_value)))} /></div>
      <Panel><div className="panel-heading"><div><div className="eyebrow">CURRENT BATCH</div><h2>Scored customers</h2></div><div className="table-actions"><label className="search-box"><Search size={15} /><input value={search} onChange={(event) => { setSearch(event.target.value); setPage(0) }} placeholder="Search customer or date" /></label><Button className="secondary" onClick={exportResults}><ArrowDownToLine size={15} />Export results</Button></div></div>
        <div className="table-wrap"><table><thead><tr>{[['CustomerID', 'Customer ID'], ['churn_probability', 'Churn probability'], ['predicted_90d_value', 'Predicted 90-day value'], ['risk_weighted_value', 'Risk-weighted value'], ['churn_rank', 'Churn rank'], ['value_rank', 'Value rank'], ['risk_value_rank', 'Risk rank']].map(([key, label]) => <th key={key}><button className="sort-button" onClick={() => { setAscending(sort === key ? !ascending : true); setSort(key as keyof ScoreResult) }}>{label}<ArrowUpDown size={12} /></button></th>)}</tr></thead><tbody>{current.map((row, index) => <tr key={`${row.CustomerID}-${index}`}><td className="customer-cell">{row.CustomerID ?? '—'}</td><td>{formatProbability(row.churn_probability)}</td><td>{formatCurrency(row.predicted_90d_value)}</td><td>{formatCurrency(row.risk_weighted_value)}</td><td>{row.churn_rank ?? '—'}</td><td>{row.value_rank ?? '—'}</td><td>{row.risk_value_rank ?? '—'}</td></tr>)}</tbody></table></div>
        <div className="table-footer"><span>Showing {filtered.length ? page * 10 + 1 : 0}–{Math.min(page * 10 + 10, filtered.length)} of {filtered.length.toLocaleString()}</span><div><Button className="secondary compact" disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</Button><Button className="secondary compact" disabled={(page + 1) * 10 >= filtered.length} onClick={() => setPage((value) => value + 1)}>Next</Button></div></div>
      </Panel>
      <Panel className="chart-panel"><div className="panel-heading"><div><div className="eyebrow">CURRENT BATCH</div><h2>Churn probability vs predicted value</h2><p>Each point is a scored customer snapshot.</p></div></div><div className="chart-box"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 12, right: 16, bottom: 20, left: 8 }}><CartesianGrid stroke="#e7e9ed" /><XAxis type="number" dataKey="predicted_90d_value" name="Predicted 90-day value" tickFormatter={(value) => `£${Math.round(value)}`} tick={{ fill: '#697386', fontSize: 11 }} /><YAxis type="number" dataKey="churn_probability" name="Churn probability" tickFormatter={(value) => `${Math.round(value * 100)}%`} domain={[0, 1]} tick={{ fill: '#697386', fontSize: 11 }} /><Tooltip cursor={{ strokeDasharray: '3 3' }} content={<ScatterTooltip />} /><Scatter data={results.map((row) => ({ ...row, name: String(row.CustomerID ?? 'Unspecified') }))} fill="#276b62" fillOpacity={0.68} /></ScatterChart></ResponsiveContainer></div><p className="fineprint">Chart summarizes only the batch currently loaded. Risk-weighted value is used for prioritization, not evidence of recoverable revenue.</p></Panel>
    </>}
  </div>
}

function Summary({ label, value }: { label: string; value: string }) { return <div className="summary-tile"><span>{label}</span><b>{value}</b></div> }
function ScatterTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ScoreResult & { name: string } }> }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  return <div className="chart-tooltip"><b>Customer {row.name}</b><span>Churn probability: {formatProbability(row.churn_probability)}</span><span>Predicted value: {formatCurrency(row.predicted_90d_value)}</span><span>Risk-weighted value: {formatCurrency(row.risk_weighted_value)}</span></div>
}

function downloadCsv(contents: string, name = 'churnguard-customer-template.csv') {
  const url = URL.createObjectURL(new Blob([contents], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a'); link.href = url; link.download = name; link.click(); URL.revokeObjectURL(url)
}

function friendlyError(error: Error): string {
  if (error instanceof ApiError) return error.status === 422 ? `The API rejected the feature values: ${error.message}` : error.message
  return 'The request could not be completed. Check the service and try again.'
}
