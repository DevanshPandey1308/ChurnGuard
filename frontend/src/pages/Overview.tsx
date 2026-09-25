import { ArrowRight, ArrowUpRight, CircleCheck, CircleHelp, Database, GitBranch, Layers3, Server } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { PageHeader, Panel, StateMessage } from '../components/ui'

export function Overview() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000, retry: 1 })
  const ready = health.data?.status === 'ok'
  return <>
    <PageHeader eyebrow="CUSTOMER INTELLIGENCE" title="ChurnGuard" description="Customer Churn & Value Intelligence" action={<a className="button primary" href="/scoring">Open customer scoring <ArrowUpRight size={15} /></a>} />
    <div className="overview-intro"><p className="lead">Predict churn risk, estimate future 90-day customer value, and prioritize customers using risk-weighted value.</p><p className="muted">Score a point-in-time feature snapshot or a validated customer batch through the ChurnGuard inference API.</p></div>

    <Panel className="system-panel"><div className="health-topline"><div><div className="eyebrow">SYSTEM STATUS</div><h2>Inference readiness</h2></div><span className={`status-pill ${ready ? 'ready' : health.isPending ? 'pending' : 'unavailable'}`}><span />{health.isPending ? 'Checking API' : ready ? 'Operational' : 'API unavailable'}</span></div>
      {health.isPending && <StateMessage kind="loading">Checking the ChurnGuard API…</StateMessage>}
      {health.isError && <div className="health-error"><StateMessage kind="error">Could not reach the API. Confirm it is running at the configured VITE_API_URL, then retry.</StateMessage><button className="text-button" onClick={() => void health.refetch()}>Retry connection</button></div>}
      {health.data && <div className="health-grid">{[['Churn model', health.data.churn_model_loaded], ['Calibration', health.data.calibrator_loaded], ['Value model', health.data.future_value_model_loaded]].map(([label, loaded]) => <div className="health-item" key={String(label)}><CircleCheck size={16} className={loaded ? 'icon-green' : 'icon-muted'} /><span>{label}</span><b>{loaded ? 'Ready' : 'Unavailable'}</b></div>)}</div>}
    </Panel>

    <Panel className="flow-panel"><div className="panel-heading"><div><div className="eyebrow">FROM SNAPSHOT TO PRIORITY</div><h2>How a score is made</h2></div><span className="flow-caption">Features are evaluated as of the selected snapshot date.</span></div><div className="workflow workflow-product">{[
      [Database, 'Feature snapshot', 'Observed customer behaviour'], [Layers3, 'Churn + value models', 'Calibrated risk and future net spend'], [GitBranch, 'Risk-weighted value', 'Combined by the inference API'], [Server, 'Customer prioritization', 'Decision support for review'],
    ].map(([Icon, title, description], index, all) => <div className="workflow-step" key={String(title)}><div className="workflow-node"><span className="workflow-number">0{index + 1}</span><Icon size={16} /><div><b>{String(title)}</b><small>{String(description)}</small></div></div>{index < all.length - 1 && <ArrowRight size={15} className="workflow-arrow" />}</div>)}</div></Panel>

    <section className="output-section"><div className="output-section-heading"><div><div className="eyebrow">PREDICTION OUTPUTS</div><h2>Three signals returned for each customer</h2></div></div><div className="output-definitions">
      <Output title="Churn Probability" description="Estimated probability of no repurchase during the next 90 days." />
      <Output title="Predicted 90-Day Value" description="Model-predicted future net spend over the following 90-day horizon; not classical CLV." />
      <Output title="Risk-Weighted Value" description="A prioritization metric combining churn probability with predicted 90-day value." />
    </div></section>
    <div className="notice"><CircleHelp size={16} /><p><strong>Decision support, not a guarantee.</strong> Risk-weighted value is not guaranteed recoverable revenue, ROI, or a measured campaign effect.</p></div>
    <div className="overview-next"><div><div className="eyebrow">NEXT STEP</div><h2>Start with a customer snapshot or batch</h2><p>Use scoring to submit engineered features and review the prediction returned by the API.</p></div><a className="button primary" href="/scoring">Go to customer scoring <ArrowRight size={15} /></a></div>
    <div className="overview-footnote">Power BI remains the separate reporting layer. This interface is for interactive scoring and model outputs.</div>
  </>
}

function Output({ title, description }: { title: string; description: string }) {
  return <article className="output-definition"><h3>{title}</h3><p>{description}</p></article>
}
