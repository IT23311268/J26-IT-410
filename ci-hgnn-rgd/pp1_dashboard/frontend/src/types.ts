export type Mode = 'real' | 'demo';
export type PageName = 'overview' | 'intents' | 'topics' | 'graph' | 'gaps' | 'validation';
export type Envelope<T> = {
  mode: Mode;
  status: 'available' | 'unavailable';
  notice: string | null;
  graph_version: string | null;
  data: T | null;
};
export type Distribution = { label: string; count: number }[];
export type Provenance = Record<string, unknown>[];
export type NodeRecord = {
  id: string; type: string; label: string; year: number | null;
  abstract: string | null; keywords: string[]; provenance: Provenance;
  metadata: Record<string, unknown>; synthetic: boolean;
};
export type EdgeRecord = {
  id: string; source: string; target: string; relation: string;
  provenance: Provenance; synthetic: boolean;
};
export type Page<T> = { items: T[]; total: number; limit: number; offset: number };
export type Stage = {
  number: number; name: string;
  status: 'Verified Data' | 'Validation Pending' | 'Not Yet Implemented';
  detail: string; evidence: string[];
};
export type Dataset = { name: string; records: number | null; status: string; detail: string; source: string | null };
export type OverviewData = {
  counts: { papers: number; authors: number; citations: number; topics: number };
  publication_years: Distribution; intent_distribution: Distribution;
  topic_coverage: { assigned: number; papers: number; fraction: number };
  graph: { nodes: number; edges: number; components: number | null };
  datasets: Dataset[]; stages: Stage[]; provenance: string[]; limitations: string[];
  intent_model: string | null; created_at: string | null;
};
export type IntentRecord = {
  id: string; text: string | null; label: 'background' | 'method' | 'result';
  confidence: number | null; source: string; split: string | null;
  citing_paper_id: string | null; cited_paper_id: string | null;
  model_id: string; model_version: string; provenance: Provenance;
  synthetic: boolean; original_label: string | null;
  source_citing_id?: string | null; source_cited_id?: string | null;
};
export type IntentData = Page<IntentRecord> & {
  distribution: Distribution; model_id: string | null; model_version: string | null; calibration: string | null;
};
export type Topic = {
  id: string; label: string; keywords: string[]; paper_count: number; coverage: number;
  model_version: string; review_status: string; synthetic: boolean;
};
export type TopicDetail = {
  topic: Topic; papers: NodeRecord[];
  evidence: { tier: string; label: string; paper_id: string; span: string | null; uncertainty: string; synthetic: boolean }[];
  provenance: Provenance; membership_note: string;
};
export type GraphStats = {
  node_counts: Record<string, number>; edge_counts: Record<string, number>;
  components: number | null; limitations: string[];
};
export type Neighborhood = { nodes: NodeRecord[]; edges: EdgeRecord[]; seed: string | null; truncated: boolean; limit: number; depth: number };
export type NodeDetail = { node: NodeRecord; relations: EdgeRecord[] };
export type ValidationData = {
  checks: { name: string; status: string; detail: string }[];
  metrics: { model: string; split: string; metric: string; value: number; samples: number | null; source: string }[];
  plans: { name: string; status: string; detail: string }[];
  missing_evidence: string[]; temporal_policy: string | null;
};
export type ProgressData = { stages: Stage[]; engineering_note: string; scientific_note: string };
