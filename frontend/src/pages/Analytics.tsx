import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { ArrowUpDown, Search } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { Button, PageHeader, Panel } from '../components/ui'
import type { ScoreResult } from '../types/api'
import { formatCurrency, formatProbability } from '../lib/features'

type SortKey = 'CustomerID' | 'risk_value_rank' | 'churn_probability' | 'predicted_90d_value' | 'risk_weighted_value'

export function Analytics() {
  const rows = useQueryClient().getQueryData<ScoreResult[]>(['batch-results']) ?? []
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState<SortKey>('risk_value_rank')
  const [ascending, setAscending] = useState(true)
  const [page, setPage] = useState(0)
  const [riskSort, setRiskSort] = useState(true)
  const filteredRows = useMemo(() => {
    const filtered = rows.filter((row) => `${row.CustomerID ?? ''} ${row.snapshot_date ?? ''}`.toLowerCase().includes(search.toLowerCase()))
    return filtered.sort((a, b) => {
      const comparison = sort === 'CustomerID'
        ? String(a.CustomerID ?? '').localeCompare(String(b.CustomerID ?? ''), undefined, { numeric: true })
        : sort === 'risk_value_rank'
        ? (a.risk_value_rank ?? Number.MAX_SAFE_INTEGER) - (b.risk_value_rank ?? Number.MAX_SAFE_INTEGER)
        : a[sort] - b[sort]
      return ascending ? comparison : -comparison
    })
  }, [rows, search, sort, ascending])
  const topRisk = useMemo(() => [...rows].sort((a, b) => riskSort
    ? b.risk_weighted_value - a.risk_weighted_value
    : b.churn_probability - a.churn_probability).slice(0, 10), [rows, riskSort])
  const visibleRows = filteredRows.slice(page * 10, page * 10 + 10)

  return <>
    <PageHeader eyebrow="INFERENCE ANALYTICS" title="Analytics" description="Explore prediction patterns and customer rankings from the currently loaded batch." />
    {rows.length === 0 ? <Panel className="analytics-empty"><div className="eyebrow">NO SCORED BATCH YET</div><h2>Score a batch to explore customer patterns</h2><p>Batch-scoped summaries and charts will appear here after the API returns predictions. No historical customer or revenue metrics are shown without scored data.</p><a className="button primary" href="/scoring">Go to customer scoring</a></Panel> : <>
      <div className="scope-note"><span className="status-dot" />Current scored batch <b>{rows.length.toLocaleString()} customers</b></div>
      <div className="section-title"><div className="eyebrow">SUMMARY</div><h2>Returned batch signals</h2></div>
      <div className="batch-kpis analytics-kpis">
        <Summary label="Customers scored" value={rows.length.toLocaleString()} />
        <Summary label="Highest churn probability" value={formatProbability(Math.max(...rows.map((row) => row.churn_probability)))} />
        <Summary label="Highest predicted 90-day value" value={formatCurrency(Math.max(...rows.map((row) => row.predicted_90d_value)))} />
        <Summary label="Highest risk-weighted value" value={formatCurrency(Math.max(...rows.map((row) => row.risk_weighted_value)))} />
      </div>
      <div className="analytics-grid analytics-workspace">
        <Panel className="chart-panel analytics-scatter"><div className="eyebrow">RISK / VALUE DISTRIBUTION</div><h2>Churn probability vs predicted 90-day value</h2><p className="muted">One point per scored customer snapshot · current batch only</p>
          <div className="chart-box"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 12, right: 18, bottom: 20, left: 10 }}><CartesianGrid stroke="#e7e9ed" /><XAxis type="number" dataKey="predicted_90d_value" name="Predicted 90-day value" tickFormatter={(value) => `£${Math.round(value)}`} tick={{ fill: '#697386', fontSize: 10 }} /><YAxis type="number" dataKey="churn_probability" name="Churn probability" tickFormatter={(value) => `${Math.round(value * 100)}%`} domain={[0, 1]} tick={{ fill: '#697386', fontSize: 10 }} /><Tooltip cursor={{ strokeDasharray: '3 3' }} content={<ScoreTooltip />} /><Scatter data={rows.map((row) => ({ ...row, name: String(row.CustomerID ?? 'Unspecified') }))} fill="#276b62" fillOpacity={0.72} /></ScatterChart></ResponsiveContainer></div>
          <p className="fineprint">The axes use model predictions. They do not estimate intervention response or revenue recovery.</p>
        </Panel>
        <Panel className="chart-panel priority-panel"><div className="eyebrow">PRIORITIZATION</div><h2>Top customers by returned value</h2><p className="muted">Sort by risk-weighted value or churn probability.</p>
          <div className="rank-switch"><button className={riskSort ? 'selected' : ''} onClick={() => setRiskSort(true)}>Risk-weighted value</button><button className={!riskSort ? 'selected' : ''} onClick={() => setRiskSort(false)}>Churn probability</button></div>
          <div className="priority-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={topRisk.map((row) => ({ ...row, name: String(row.CustomerID ?? '—'), chart_value: riskSort ? row.risk_weighted_value : row.churn_probability }))} layout="vertical" margin={{ top: 4, right: 17, bottom: 4, left: 5 }}><CartesianGrid stroke="#e7e9ed" horizontal={false} /><XAxis type="number" hide /><YAxis type="category" dataKey="name" width={84} tick={{ fill: '#56616b', fontSize: 10 }} /><Tooltip content={<PriorityTooltip riskSort={riskSort} />} /><Bar dataKey="chart_value" radius={[0, 3, 3, 0]} barSize={13}>{topRisk.map((row, index) => <Cell key={`${row.CustomerID}-${index}`} fill={index === 0 ? '#276b62' : '#77a89e'} />)}</Bar></BarChart></ResponsiveContainer></div>
        </Panel>
      </div>
      <Panel className="analytics-table-panel"><div className="panel-heading"><div><div className="eyebrow">CUSTOMER RESULTS</div><h2>Scored customer snapshots</h2></div><label className="search-box"><Search size={15} /><input aria-label="Search customer results" value={search} onChange={(event) => { setSearch(event.target.value); setPage(0) }} placeholder="Search ID or date" /></label></div>
        <div className="table-wrap"><table className="data-table"><thead><tr><th><SortButton label="CustomerID" active={sort === 'CustomerID'} onClick={() => { setAscending(sort === 'CustomerID' ? !ascending : true); setSort('CustomerID') }} /></th><th><SortButton label="Churn Probability" active={sort === 'churn_probability'} onClick={() => { setAscending(sort === 'churn_probability' ? !ascending : false); setSort('churn_probability') }} /></th><th><SortButton label="Predicted 90-Day Value" active={sort === 'predicted_90d_value'} onClick={() => { setAscending(sort === 'predicted_90d_value' ? !ascending : false); setSort('predicted_90d_value') }} /></th><th><SortButton label="Risk-Weighted Value" active={sort === 'risk_weighted_value'} onClick={() => { setAscending(sort === 'risk_weighted_value' ? !ascending : false); setSort('risk_weighted_value') }} /></th><th>Churn Rank</th><th>Value Rank</th><th>Risk-Value Rank</th></tr></thead><tbody>{visibleRows.map((row, index) => <tr key={`${row.CustomerID}-${index}`}><td className="customer-cell">{row.CustomerID ?? '—'}</td><td>{formatProbability(row.churn_probability)}</td><td>{formatCurrency(row.predicted_90d_value)}</td><td>{formatCurrency(row.risk_weighted_value)}</td><td>{row.churn_rank ?? '—'}</td><td>{row.value_rank ?? '—'}</td><td>{row.risk_value_rank ?? '—'}</td></tr>)}</tbody></table></div>
        <div className="table-footer"><span>Showing {filteredRows.length ? page * 10 + 1 : 0}–{Math.min(page * 10 + 10, filteredRows.length)} of {filteredRows.length.toLocaleString()}</span><div><Button className="secondary compact" disabled={page === 0} onClick={() => setPage((value) => Math.max(0, value - 1))}>Previous</Button><Button className="secondary compact" disabled={(page + 1) * 10 >= filteredRows.length} onClick={() => setPage((value) => value + 1)}>Next</Button></div></div>
      </Panel>
    </>}
  </>
}

function Summary({ label, value }: { label: string; value: string }) { return <div className="summary-tile"><span>{label}</span><b>{value}</b></div> }
function SortButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) { return <button className={`sort-button ${active ? 'sorted' : ''}`} onClick={onClick}>{label}<ArrowUpDown size={12} /></button> }
function ScoreTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ScoreResult & { name: string } }> }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  return <div className="chart-tooltip"><b>Customer {row.name}</b><span>Churn probability: {formatProbability(row.churn_probability)}</span><span>Predicted 90-day value: {formatCurrency(row.predicted_90d_value)}</span><span>Risk-weighted value: {formatCurrency(row.risk_weighted_value)}</span></div>
}
function PriorityTooltip({ active, payload, riskSort }: { active?: boolean; payload?: Array<{ payload: ScoreResult & { name: string } }>; riskSort: boolean }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  return <div className="chart-tooltip"><b>Customer {row.name}</b><span>{riskSort ? `Risk-weighted value: ${formatCurrency(row.risk_weighted_value)}` : `Churn probability: ${formatProbability(row.churn_probability)}`}</span><span>Predicted 90-day value: {formatCurrency(row.predicted_90d_value)}</span></div>
}
