import { Info, Layers3 } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PageHeader, Panel } from '../components/ui'
import shapImportance from '../data/global_feature_importance.json'

type ImportanceRow = { feature: string; mean_abs_shap: number; rank: number }
const importance = shapImportance as ImportanceRow[]

export function Explainability() {
  return <>
    <PageHeader eyebrow="MODEL INTERPRETATION" title="Explainability" description="Review the existing global SHAP analysis and understand what it can—and cannot—tell you." />
    <Panel className="shap-panel"><div className="shap-heading"><div className="shap-icon"><Layers3 size={18} /></div><div><div className="eyebrow">GLOBAL MODEL EXPLAINABILITY</div><h2>Features with the largest mean contribution magnitude</h2><p>Top 10 features by mean absolute SHAP value from the existing offline churn-model analysis.</p></div><span className="artifact-tag">Verified project artifact</span></div>
      <div className="shap-chart" role="img" aria-label="Horizontal bar chart showing mean absolute SHAP values for the top ten model features"><ResponsiveContainer width="100%" height="100%"><BarChart data={importance} layout="vertical" margin={{ top: 6, right: 30, bottom: 10, left: 20 }}><CartesianGrid stroke="#edf0f1" horizontal={false} /><XAxis type="number" domain={[0, 'dataMax']} tick={{ fill: '#7a858d', fontSize: 10 }} tickFormatter={(value) => Number(value).toFixed(2)} /><YAxis type="category" dataKey="feature" width={190} tick={{ fill: '#46535a', fontSize: 10 }} /><Tooltip content={<ShapTooltip />} /><Bar dataKey="mean_abs_shap" fill="#397d72" radius={[0, 3, 3, 0]} barSize={17} /></BarChart></ResponsiveContainer></div>
      <div className="shap-table-wrap"><table className="shap-table"><thead><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP| · raw margin</th></tr></thead><tbody>{importance.map((row) => <tr key={row.feature}><td>{String(row.rank).padStart(2, '0')}</td><td><code>{row.feature}</code></td><td>{row.mean_abs_shap.toFixed(4)}</td></tr>)}</tbody></table></div>
      <div className="shap-caption">Source: <code>artifacts/shap/global_feature_importance.csv</code> from the existing June held-out feature analysis. Values are mean absolute contributions in raw model-margin space—not probability points. The frontend copy preserves the source ranking and values.</div>
    </Panel>
    <div className="notice"><Info size={17} /><p><strong>Global, not customer-specific.</strong> Global SHAP describes model-wide feature contribution patterns across the analyzed June snapshots. It does not explain an individual customer's prediction. The inference API does not expose per-customer SHAP values.</p></div>
    <div className="explain-grid"><Panel><div className="eyebrow">HOW TO READ THIS</div><h2>Magnitude does not show direction</h2><p className="panel-copy">Mean absolute SHAP ranks the typical size of a feature's contribution to the model output. It does not indicate whether a higher feature value raises or lowers churn probability. Correlated features can share contribution.</p></Panel><Panel><div className="eyebrow">ANALYSIS SCOPE</div><h2>Existing model, held-out snapshots</h2><p className="panel-copy">The analysis explains the fitted LightGBM churn model on June feature rows. June labels are not used to compute SHAP values. These are learned model associations, not causes or treatment effects.</p></Panel></div>
  </>
}

function ShapTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: ImportanceRow }> }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  return <div className="chart-tooltip"><b>{row.feature}</b><span>Mean absolute SHAP: {row.mean_abs_shap.toFixed(4)}</span><span>Rank: {row.rank}</span></div>
}
