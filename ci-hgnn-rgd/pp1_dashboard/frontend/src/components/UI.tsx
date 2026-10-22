import { AlertCircle, ArrowRight, Check, ChevronLeft, ChevronRight, Database, Info, LoaderCircle, Search, ShieldCheck } from 'lucide-react';
import type { ReactNode } from 'react';
import type { RequestState } from '../api';
import type { Mode, Provenance } from '../types';

export const SYNTHETIC_LABEL = 'Synthetic Demo Data — Not Research Findings';
export const formatNumber = (value: number | null | undefined) => value == null ? 'Unavailable' : new Intl.NumberFormat('en-US').format(value);
export const percent = (value: number | null | undefined) => value == null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;
export const humanize = (value: string) => value.replaceAll('_', ' ');

export function Synthetic({ mode, synthetic }: { mode?: Mode; synthetic?: boolean }) {
  return mode === 'demo' || synthetic ? <span className="synthetic-label" data-testid="synthetic-label">{SYNTHETIC_LABEL}</span> : null;
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'teal' | 'blue' | 'amber' | 'neutral' | 'purple' }) {
  return <span className={`badge badge-${tone}`}><span className="badge-dot" />{children}</span>;
}

export function StatusBadge({ status }: { status: string }) {
  const lower = status.toLowerCase();
  const tone = lower.includes('verified') || lower === 'passed' || lower === 'available' ? 'teal' : lower.includes('prediction') ? 'blue' : lower.includes('pending') || lower.includes('unavailable') ? 'amber' : 'neutral';
  return <Badge tone={tone}>{humanize(status)}</Badge>;
}

export function PageHeader({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children?: ReactNode }) {
  return <header className="page-header"><div><div className="eyebrow">{eyebrow}</div><h1 tabIndex={-1}>{title}</h1><p>{description}</p></div>{children && <div className="header-actions">{children}</div>}</header>;
}

export function SectionTitle({ title, subtitle, aside }: { title: string; subtitle?: string; aside?: ReactNode }) {
  return <div className="section-title"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{aside}</div>;
}

export function ApiBoundary<T>({ state, children, compact = false }: { state: RequestState<T>; children: (data: T) => ReactNode; compact?: boolean }) {
  if (state.loading) return <div className={`state-panel ${compact ? 'compact' : ''}`} role="status"><LoaderCircle className="spin" size={25} /><h3>Reading published artifacts</h3><p>Loading the selected evidence source.</p></div>;
  if (state.error) return <div className={`state-panel state-error ${compact ? 'compact' : ''}`} role="alert"><AlertCircle size={29} /><h3>Unable to load this view</h3><p>{state.error} Check that the dashboard backend is running.</p><button className="button secondary" onClick={state.reload}>Try again <ArrowRight size={15} /></button></div>;
  if (!state.response || state.response.status === 'unavailable' || state.response.data === null) return <div className={`state-panel ${compact ? 'compact' : ''}`} data-testid="unavailable-state"><Database size={30} /><h3>Artifact unavailable</h3><p>{state.response?.notice ?? 'No verified artifact is available for this view.'}</p><span className="subtle">Real artifacts remain selected. Demo mode requires an explicit switch.</span></div>;
  return <>{state.response.notice && state.response.mode === 'real' && <Notice>{state.response.notice}</Notice>}{children(state.response.data)}</>;
}

export function Notice({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'amber' | 'teal' }) {
  return <div className={`notice notice-${tone}`}><Info size={17} /><div>{children}</div></div>;
}

export function SearchInput({ value, onChange, placeholder, label }: { value: string; onChange: (value: string) => void; placeholder: string; label: string }) {
  return <label className="search-field"><Search size={17} aria-hidden="true" /><span className="sr-only">{label}</span><input type="search" value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} /></label>;
}

export function Pagination({ total, limit, offset, onOffset }: { total: number; limit: number; offset: number; onOffset: (offset: number) => void }) {
  return <div className="pagination"><span>{total ? `${formatNumber(offset + 1)}–${formatNumber(Math.min(offset + limit, total))} of ${formatNumber(total)}` : 'No matching records'}</span><div><button className="icon-button" aria-label="Previous page" disabled={offset <= 0} onClick={() => onOffset(Math.max(0, offset - limit))}><ChevronLeft size={17} /></button><button className="icon-button" aria-label="Next page" disabled={offset + limit >= total} onClick={() => onOffset(offset + limit)}><ChevronRight size={17} /></button></div></div>;
}

export function ProvenanceDetails({ value, title = 'Source provenance' }: { value: Provenance | string[] | Record<string, unknown>; title?: string }) {
  return <details className="provenance"><summary><ShieldCheck size={15} />{title}</summary><pre>{JSON.stringify(value, null, 2)}</pre></details>;
}

export function Empty({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-inline"><Search size={22} /><h3>{title}</h3><p>{detail}</p></div>;
}

export function MiniCheck({ children }: { children: ReactNode }) { return <span className="mini-check"><Check size={13} />{children}</span>; }
