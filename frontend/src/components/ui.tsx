import type { ButtonHTMLAttributes, PropsWithChildren, ReactNode } from 'react'
import { AlertCircle, CheckCircle2, LoaderCircle } from 'lucide-react'

export function Button({ children, className = '', ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`button ${className}`} {...props}>{children}</button>
}

export function PageHeader({ eyebrow, title, description, action }: { eyebrow?: string; title: string; description?: string; action?: ReactNode }) {
  return <header className="page-header"><div>{eyebrow && <div className="eyebrow">{eyebrow}</div>}<h1>{title}</h1>{description && <p>{description}</p>}</div>{action && <div className="page-header-action">{action}</div>}</header>
}

export function Panel({ children, className = '' }: PropsWithChildren<{ className?: string }>) {
  return <section className={`panel ${className}`}>{children}</section>
}

export function StateMessage({ kind, children }: PropsWithChildren<{ kind: 'error' | 'success' | 'loading' }>) {
  const Icon = kind === 'error' ? AlertCircle : kind === 'success' ? CheckCircle2 : LoaderCircle
  return <div className={`state-message ${kind}`} role={kind === 'error' ? 'alert' : 'status'}><Icon size={17} className={kind === 'loading' ? 'spin' : ''} />{children}</div>
}

export function FieldLabel({ name, label, hint }: { name: string; label?: string; hint?: string }) {
  return <span className="field-label">{label || name.replaceAll('_', ' ')}{hint && <small>{hint}</small>}</span>
}
