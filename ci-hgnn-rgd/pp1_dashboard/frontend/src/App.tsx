import { Activity, ArrowUpRight, Atom, BookOpen, ChevronRight, Compass, FlaskConical, Layers3, LayoutDashboard, Menu, Network, PanelLeftClose, ScanSearch, ShieldCheck, X } from 'lucide-react';
import { Suspense, lazy, useEffect, useState } from 'react';
import type { Mode, PageName } from './types';
import { Overview } from './pages/Overview';
import { Intents } from './pages/Intents';
import { Topics } from './pages/Topics';
import { Gaps } from './pages/Gaps';
import { Validation } from './pages/Validation';

const Graph = lazy(() => import('./pages/Graph'));
const navigation = [
  { id: 'overview', label: 'Research overview', icon: LayoutDashboard, section: 'Explore' },
  { id: 'intents', label: 'Citation intents', icon: BookOpen, section: 'Explore' },
  { id: 'topics', label: 'Scientific topics', icon: Layers3, section: 'Explore' },
  { id: 'graph', label: 'Knowledge graph', icon: Network, section: 'Explore' },
  { id: 'gaps', label: 'Research gaps', icon: ScanSearch, section: 'Research' },
  { id: 'validation', label: 'Validation & progress', icon: ShieldCheck, section: 'Research' },
] as const;

function pageFromHash(): PageName {
  const candidate = window.location.hash.replace('#', '').split('?')[0];
  return navigation.some((item) => item.id === candidate) ? candidate as PageName : 'overview';
}

export default function App() {
  const [page, setPage] = useState<PageName>(pageFromHash);
  const [mode, setMode] = useState<Mode>('real');
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    const change = () => { setPage(pageFromHash()); setMenuOpen(false); window.scrollTo({ top: 0 }); };
    window.addEventListener('hashchange', change);
    return () => window.removeEventListener('hashchange', change);
  }, []);
  useEffect(() => {
    document.title = `${navigation.find((item) => item.id === page)?.label ?? 'Research Intelligence'} · CI-HGNN-RGD`;
  }, [page]);
  useEffect(() => {
    if (!menuOpen) return;
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setMenuOpen(false); };
    window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [menuOpen]);
  return <div className={`app-shell ${mode === 'demo' ? 'demo-mode' : ''}`}>
    <a href="#main-content" className="skip-link" onClick={(event) => { event.preventDefault(); document.getElementById('main-content')?.focus(); }}>Skip to content</a>
    {menuOpen && <button className="nav-backdrop" aria-label="Close navigation" onClick={() => setMenuOpen(false)} />}
    <aside className={`sidebar ${menuOpen ? 'is-open' : ''}`}>
      <a className="brand" href="#overview" aria-label="CI-HGNN-RGD home"><span className="brand-mark"><Atom size={26} strokeWidth={1.6} /></span><span><strong>CI-HGNN-RGD<span className="brand-period">.</span></strong><small>Research Gap Intelligence</small></span></a>
      <button className="mobile-close icon-button" aria-label="Close navigation" onClick={() => setMenuOpen(false)}><X size={20} /></button>
      <div className="workspace-label"><span className="status-pulse" />RESEARCH WORKSPACE<span className="version-pill">PP1</span></div>
      <nav aria-label="Main navigation">{['Explore', 'Research'].map((section) => <div key={section} className="nav-section"><div className="nav-caption">{section}</div>{navigation.filter((item) => item.section === section).map(({ id, label, icon: Icon }) => <a key={id} href={`#${id}`} className={`nav-link ${page === id ? 'active' : ''}`} aria-current={page === id ? 'page' : undefined} data-testid={`nav-${id}`}><Icon size={18} /><span>{label}</span>{id === 'gaps' ? <span className="nav-preview">Preview</span> : page === id && <ChevronRight size={14} className="nav-chevron" />}</a>)}</div>)}</nav>
      <div className="sidebar-bottom"><div className="research-note"><FlaskConical size={22} /><div className="research-note-title">Evidence before insight.</div><p>Trace the literature.<br />Understand the connections.<br />Keep uncertainty visible.</p><span>ISOLATED PP1 PROTOTYPE <ArrowUpRight size={13} /></span></div><div className="sidebar-footer"><div className="avatar">RG</div><div><strong>Research workspace</strong><small>CI-HGNN-RGD · Stage 06</small></div><PanelLeftClose size={17} /></div></div>
    </aside>
    <div className="workspace">
      <div className="topbar"><div className="breadcrumbs"><button className="mobile-toggle icon-button" aria-label="Open navigation" onClick={() => setMenuOpen(true)}><Menu size={21} /></button><Compass size={17} /><span>Workspace</span><ChevronRight size={13} /><strong>{navigation.find((item) => item.id === page)?.label}</strong></div><div className="topbar-right"><span className="read-only"><ShieldCheck size={14} /> Read-only artifacts</span><div className="mode-switch" role="group" aria-label="Data source mode"><button aria-pressed={mode === 'real'} onClick={() => setMode('real')} data-testid="mode-real">Real data</button><button aria-pressed={mode === 'demo'} onClick={() => setMode('demo')} data-testid="mode-demo">Demo</button></div></div></div>
      {mode === 'demo' && <div className="demo-banner" role="status" data-testid="demo-banner"><FlaskConical size={17} /><strong>DEMO MODE — SYNTHETIC DATA — NOT RESEARCH RESULTS</strong><span>Interface demonstration only</span></div>}
      <main id="main-content" tabIndex={-1} key={`${page}:${mode}`}>
        <Suspense fallback={<div className="state-panel" role="status"><Activity className="spin" /><h3>Opening the graph workspace</h3></div>}>
          {page === 'overview' && <Overview mode={mode} />}
          {page === 'intents' && <Intents mode={mode} />}
          {page === 'topics' && <Topics mode={mode} />}
          {page === 'graph' && <Graph mode={mode} />}
          {page === 'gaps' && <Gaps mode={mode} />}
          {page === 'validation' && <Validation mode={mode} />}
        </Suspense>
        <footer className="page-footer"><span>CI-HGNN-RGD <span className="footer-separator">/</span> Research Gap Intelligence</span><span><span className="tiny-dot" /> {mode === 'real' ? 'Published evidence · Read-only' : 'Synthetic interface demonstration'}</span></footer>
      </main>
    </div>
  </div>;
}
