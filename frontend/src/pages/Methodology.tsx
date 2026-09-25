import { PageHeader, Panel } from '../components/ui'

const sections = [
  ['Problem', 'Estimate whether a customer will not repurchase within the next 90 days and predict net spend in the same future window.'],
  ['Point-in-time feature design', 'Repeated customer snapshots use transactions strictly before each snapshot date. Future labels use the following (t, t + 90 days] window.'],
  ['Churn modeling', 'A LightGBM binary classifier estimates churn probability. The probability output is calibrated with sigmoid (Platt) calibration.'],
  ['Future 90-day value', 'A LightGBM regression model with a signed-log target transformation predicts future 90-day net spend. This is not classical customer lifetime value.'],
  ['Risk-weighted targeting', 'The API returns churn probability × predicted future 90-day value as a prioritization signal. It does not estimate causal treatment effects or guaranteed recovered revenue.'],
  ['Temporal evaluation', 'Training, validation, and final test use chronological snapshot dates. Random row splits are avoided because a customer can appear in multiple snapshots.'],
  ['Walk-forward validation', 'Additional temporal checks assess performance across successive historical cutoffs; future windows can overlap across snapshots.'],
  ['Explainability', 'SHAP analysis is part of the offline model workflow. The current inference API does not expose per-customer explanation values.'],
  ['Classical CLV benchmark', 'BG/NBD + Gamma-Gamma is a separate classical benchmark estimating gross purchase value. Its target differs from supervised future net spend, so the metrics are not directly interchangeable.'],
]

export function Methodology() {
  return <>
    <PageHeader eyebrow="MODEL DOCUMENTATION" title="Methodology" description="A concise guide to the current data and inference design, including where results should be interpreted cautiously." />
    <div className="method-list">{sections.map(([title, description], i) => <Panel className="method-item" key={title}><div className="method-index">{String(i + 1).padStart(2, '0')}</div><div><h2>{title}</h2><p>{description}</p></div></Panel>)}</div>
    <Panel className="limitations"><div className="eyebrow">LIMITATIONS</div><h2>Scope and caveats</h2><ul className="clean-list"><li>One retailer and historical transaction data; no customer demographics.</li><li>No campaign treatment-response data, causal inference, or uplift modeling.</li><li>Future windows overlap between customer snapshots, and only a limited number of temporal snapshots are available.</li><li>Predicted 90-day net spend is a short-horizon supervised target, not true lifetime value.</li></ul></Panel>
  </>
}
