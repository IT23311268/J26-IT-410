# PP1 API contract

GET-only API. Every endpoint except `/api/health` accepts `mode=real|demo`
(default real). Never automatically fall back to demo. Envelope:

```
{ mode, status: "available"|"unavailable", notice: string|null,
  graph_version: string|null, data: T|null }
```

Unavailable artifacts return HTTP 200 with unavailable envelope. Invalid
parameters return 422, unknown IDs 404. All frontend views handle these states.
Demo envelope notice and every demo record carry
`Synthetic Demo Data — Not Research Findings`. Demo UI persistent banner:
`DEMO MODE — SYNTHETIC DATA — NOT RESEARCH RESULTS`.

Shared records (extra provenance fields may exist):

- Node: `{id, type, label, year: number|null, abstract: string|null,
  keywords: string[], provenance: object[], metadata: object, synthetic: boolean}`.
- Edge: `{id, source, target, relation, provenance: object[], synthetic: boolean}`.
- Page: `{items: T[], total, limit, offset}`. limit 1..100; offset 0..100000.
- Distribution: `{label: string, count: number}[]`.
- Stage: `{number, name, status: "Verified Data"|"Validation Pending"|
  "Not Yet Implemented", detail, evidence: string[]}`. This is artifact status,
  never a percentage of research completion.

Endpoints and data payloads:

- `/api/health`: `{status:"ok", read_only:true, default_mode:"real"}`.
- `/api/overview`: `{counts:{papers,authors,citations,topics},
  publication_years:Distribution, intent_distribution:Distribution,
  topic_coverage:{assigned,papers,fraction}, graph:{nodes,edges,components},
  datasets:Dataset[], stages:Stage[], provenance:string[], limitations:string[],
  intent_model:string|null, created_at:string|null}`.
- `/api/datasets`: `{items: Dataset[]}`. Dataset =
  `{name,records:number|null,status,detail,source:string|null}`.
- `/api/citation-intents?q=&label=&limit=20&offset=0`:
  `{...Page, distribution:Distribution, model_id:string|null,
  model_version:string|null, calibration:string|null}`.
  Items: `{id,text:string|null,label,confidence:number|null,source,
  split:string|null,citing_paper_id:string|null,cited_paper_id:string|null,
  model_id,model_version,provenance:object[],synthetic:boolean,
  original_label:string|null}`. Label restricted to background/method/result.
- `/api/topics?q=&limit=20&offset=0`: Page of Topic.
  Topic = `{id,label,keywords:string[],paper_count,coverage:number,
  model_version,review_status,synthetic:boolean}`.
- `/api/topics/{topic_id}`: `{topic:Topic,papers:Node[],
  evidence:{tier,label,paper_id,span:string|null,uncertainty,synthetic:boolean}[],
  provenance:object[],membership_note:string}`. Max 20 papers/evidence each.
- `/api/graph/stats`: `{node_counts:Record<string,number>,
  edge_counts:Record<string,number>,components:number|null,
  limitations:string[]}`. Derived reverse edges excluded from browser counts.
- `/api/graph/nodes?q=&node_type=&limit=20&offset=0`: Page of Node.
- `/api/graph/nodes/{node_id}`: `{node:Node,relations:Edge[]}` max 50 relations.
- `/api/graph/neighborhood?seed=&depth=1&limit=70&node_types=paper,author,topic&edge_types=authored_by,associated_with_topic`:
  `{nodes:Node[],edges:Edge[],seed:string|null,truncated:boolean,
  limit:number,depth:number}`. Depth 1..2, nodes 1..150, edges max 400.
  With no seed choose a real topic that has members. Filters apply server-side;
  absent filters mean all available supported types. Empty filter string means
  none, never silently all. A seed excluded by type filters yields empty nodes.
- `/api/progress`: `{stages:Stage[],engineering_note,scientific_note}`.
- `/api/validation/status`: `{checks:{name,status,detail}[],
  metrics:{model,split,metric,value:number,samples:number|null,source}[],
  plans:{name,status,detail}[],missing_evidence:string[],temporal_policy:string|null}`.
  Metrics only from verified published model outputs; no demo numeric metrics.

Default backend port 8000, frontend 5173, Vite proxy `/api` to localhost:8000.
No upstream imports. Backend factory `create_app(component_root:Path|None=None)`
allows tests to point at isolated fixtures. Default root is inferred from the
backend file location. Optional environment file is frontend-local only;
no global environment variable configuration required.

The optional field relation preserves Stage 06's `classified_in` identifier.
Node and relation filters never rename membership to mentions/investigation.
Published immutable inputs are checksum-verified when the reader is initialized;
subsequent mtime/size changes return unavailable until restart. Typed Pydantic
response models are exposed in `/openapi.json` for each envelope payload.
