import { useEffect, useState } from 'react';
import type { Envelope, Mode } from './types';

export type RequestState<T> = {
  loading: boolean; error: string | null; response: Envelope<T> | null; reload: () => void;
};

export function useApi<T>(path: string | null, mode: Mode): RequestState<T> {
  const [state, setState] = useState<{ key: string; loading: boolean; error: string | null; response: Envelope<T> | null }>({ key: '', loading: true, error: null, response: null });
  const [revision, setRevision] = useState(0);
  const key = `${mode}:${path ?? ''}:${revision}`;
  useEffect(() => {
    if (!path) { setState({ key, loading: false, error: null, response: null }); return; }
    const controller = new AbortController();
    setState({ key, loading: true, error: null, response: null });
    const separator = path.includes('?') ? '&' : '?';
    fetch(`${path}${separator}mode=${mode}`, { signal: controller.signal, headers: { Accept: 'application/json' } })
      .then(async (response) => {
        if (!response.ok) throw new Error(response.status === 404 ? 'This record is unavailable in the selected artifact.' : `The read-only API returned HTTP ${response.status}.`);
        const payload: Envelope<T> = await response.json();
        if (payload.mode !== mode || !['available', 'unavailable'].includes(payload.status)) throw new Error('The API returned an unexpected data contract.');
        if (mode === 'real' && containsSynthetic(payload.data)) throw new Error('Synthetic records were rejected in real artifact mode.');
        return payload;
      })
      .then((response) => { if (!controller.signal.aborted) setState({ key, loading: false, error: null, response }); })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setState({ key, loading: false, error: error instanceof Error ? error.message : 'Unable to read the artifact API.', response: null });
      });
    return () => controller.abort();
  }, [path, mode, revision, key]);
  // Never render a previous mode, query or record while its replacement is loading.
  return { ...(state.key === key ? state : { loading: !!path, error: null, response: null }), reload: () => setRevision((value) => value + 1) };
}

function containsSynthetic(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  if (Array.isArray(value)) return value.some(containsSynthetic);
  const record = value as Record<string, unknown>;
  return record.synthetic === true || Object.values(record).some(containsSynthetic);
}

export function useDebounce(value: string, delay = 260): string {
  const [settled, setSettled] = useState(value);
  useEffect(() => { const timer = window.setTimeout(() => setSettled(value), delay); return () => window.clearTimeout(timer); }, [value, delay]);
  return settled;
}

export function query(parameters: Record<string, string | number>): string {
  return new URLSearchParams(Object.entries(parameters).map(([key, value]) => [key, String(value)])).toString();
}
