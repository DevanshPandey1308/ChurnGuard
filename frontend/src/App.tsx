import { lazy, Suspense, useState } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { Activity, BarChart3, BookOpenText, ChevronLeft, CircleHelp, Menu, ShieldCheck, Sparkles, UsersRound, X } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import { api } from './api/client'

const Overview = lazy(() => import('./pages/Overview').then((module) => ({ default: module.Overview })))
const Scoring = lazy(() => import('./pages/Scoring').then((module) => ({ default: module.Scoring })))
const Analytics = lazy(() => import('./pages/Analytics').then((module) => ({ default: module.Analytics })))
const Explainability = lazy(() => import('./pages/Explainability').then((module) => ({ default: module.Explainability })))
const Methodology = lazy(() => import('./pages/Methodology').then((module) => ({ default: module.Methodology })))

const navigation = [
  { label: 'Overview', to: '/', icon: BarChart3, end: true },
  { label: 'Customer scoring', to: '/scoring', icon: UsersRound },
  { label: 'Analytics', to: '/analytics', icon: Activity },
  { label: 'Explainability', to: '/explainability', icon: Sparkles },
  { label: 'Methodology', to: '/methodology', icon: BookOpenText },
]

export default function App() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000, retry: 1 })
  const location = useLocation()
  const active = navigation.find((item) => item.to !== '/' && location.pathname.startsWith(item.to))?.label ?? 'Overview'
  return <div className="app-shell">
    {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`}>
      <Link to="/" className="brand" onClick={() => setMobileOpen(false)}><span className="brand-mark"><ShieldCheck size={19} /></span><span><b>ChurnGuard</b><small>Customer intelligence</small></span></Link>
      <div className="nav-label">WORKSPACE</div><nav>{navigation.map(({ label, to, icon: Icon, end }) => <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} onClick={() => setMobileOpen(false)}><Icon size={17} /><span>{label}</span></NavLink>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-health"><span className={`health-led ${health.data?.status === 'ok' ? 'online' : ''}`} /><div><b>{health.data?.status === 'ok' ? 'API connected' : health.isPending ? 'Checking API' : 'API unavailable'}</b><small>Inference service</small></div></div><div className="sidebar-caption">90-day churn · future value</div></div>
    </aside>
    <main className="main-area"><div className="topbar"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={20} /></button><div className="breadcrumb">Workspace <ChevronLeft size={13} className="breadcrumb-chevron" /> <b>{active}</b></div><div className="topbar-status"><span className={`health-led ${health.data?.status === 'ok' ? 'online' : ''}`} />{health.data?.status === 'ok' ? 'API operational' : health.isPending ? 'Checking API…' : 'API unavailable'}</div><button className="help-button" aria-label="Methodology information" onClick={() => { window.location.href = '/methodology' }}><CircleHelp size={17} /></button></div>
      <div className="page-container"><Suspense fallback={<div className="panel" role="status">Loading workspace…</div>}><Routes><Route path="/" element={<Overview />} /><Route path="/scoring" element={<Scoring />} /><Route path="/analytics" element={<Analytics />} /><Route path="/explainability" element={<Explainability />} /><Route path="/methodology" element={<Methodology />} /><Route path="*" element={<Navigate to="/" replace />} /></Routes></Suspense></div>
    </main>
    <button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button>
  </div>
}
