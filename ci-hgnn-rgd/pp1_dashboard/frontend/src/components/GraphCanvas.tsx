import cytoscape from 'cytoscape';
import type { Core } from 'cytoscape';
import { Maximize2, Minus, Plus, RotateCcw } from 'lucide-react';
import { useEffect, useRef } from 'react';
import type { Neighborhood, NodeRecord } from '../types';

export const nodeColors: Record<string, string> = { paper: '#3886d7', author: '#9b82cc', topic: '#17aa9a', research_field: '#e3a74a' };
export const relationLabel = (relation: string) => ({ associated_with_topic: 'Topic membership', authored_by: 'Authored by', cites: 'Cites', mentions_topic: 'Mentions topic', investigates_topic: 'Investigates topic', classified_in: 'Research field' })[relation] ?? relation;

export function GraphCanvas({ data, onSelect, compact = false }: { data: Neighborhood; onSelect: (node: NodeRecord) => void; compact?: boolean }) {
  const container = useRef<HTMLDivElement>(null);
  const core = useRef<Core | null>(null);
  const callback = useRef(onSelect);
  callback.current = onSelect;
  useEffect(() => {
    if (!container.current) return;
    const graph = cytoscape({
      container: container.current,
      elements: [
        ...data.nodes.map((node) => ({ data: { id: node.id, label: node.label.length > 35 ? `${node.label.slice(0, 32)}…` : node.label, type: node.type, color: nodeColors[node.type] ?? '#7c8da8', size: node.type === 'topic' ? 33 : node.type === 'paper' ? 15 : 10 } })),
        ...data.edges.map((edge) => ({ data: { id: edge.id, source: edge.source, target: edge.target, relation: edge.relation } })),
      ],
      style: [
        { selector: 'node', style: { 'background-color': 'data(color)', width: 'data(size)', height: 'data(size)', label: compact ? '' : 'data(label)', color: '#64738a', 'font-size': 9, 'text-valign': 'bottom', 'text-margin-y': 7, 'text-outline-color': '#f9fbfd', 'text-outline-width': 2, 'border-width': 3, 'border-color': '#ffffff', 'min-zoomed-font-size': 8 } },
        { selector: 'node[type="author"]', style: { label: '' } },
        { selector: 'edge', style: { width: 1, 'line-color': '#cfdae7', 'target-arrow-color': '#b4c5d8', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.55, 'curve-style': 'bezier', opacity: 0.7 } },
        { selector: 'node:selected', style: { 'border-color': '#0d2842', 'border-width': 4, label: 'data(label)', 'font-weight': 'bold', color: '#142c44' } },
        { selector: '.muted', style: { opacity: 0.18 } },
        { selector: 'edge.highlighted', style: { 'line-color': '#158f9c', width: 2, opacity: 1 } },
      ],
      layout: { name: 'concentric', concentric: (node) => node.id() === data.seed ? 4 : node.data('type') === 'paper' ? 2 : 1, levelWidth: () => 1, minNodeSpacing: compact ? 12 : 25, padding: compact ? 25 : 45, animate: false },
      minZoom: 0.15, maxZoom: 4, wheelSensitivity: 0.2,
    });
    core.current = graph;
    graph.on('tap', 'node', (event) => {
      const selected = event.target;
      graph.elements().removeClass('muted highlighted');
      graph.elements().not(selected.closedNeighborhood()).addClass('muted');
      selected.connectedEdges().addClass('highlighted');
      const record = data.nodes.find((node) => node.id === selected.id());
      if (record) callback.current(record);
    });
    graph.on('tap', (event) => { if (event.target === graph) graph.elements().removeClass('muted highlighted'); });
    const resize = new ResizeObserver(() => { graph.resize(); graph.fit(undefined, compact ? 25 : 45); });
    resize.observe(container.current);
    return () => { resize.disconnect(); graph.destroy(); core.current = null; };
  }, [data, compact]);
  const zoom = (factor: number) => { const graph = core.current; if (graph) graph.zoom({ level: graph.zoom() * factor, renderedPosition: { x: graph.width() / 2, y: graph.height() / 2 } }); };
  return <div className={`graph-canvas-wrap ${compact ? 'compact-graph' : ''}`}>
    <div ref={container} className="graph-canvas" role="img" aria-label={`Interactive scholarly graph with ${data.nodes.length} nodes and ${data.edges.length} relationships. Select a node using the adjacent list for keyboard access.`} data-testid="graph-canvas" data-node-count={data.nodes.length} />
    <div className="graph-controls"><button className="icon-button" aria-label="Zoom in" onClick={() => zoom(1.3)}><Plus size={17} /></button><button className="icon-button" aria-label="Zoom out" onClick={() => zoom(0.77)}><Minus size={17} /></button><button className="icon-button" aria-label="Fit graph" onClick={() => core.current?.fit(undefined, 40)}><Maximize2 size={16} /></button><button className="icon-button" aria-label="Reset graph selection" onClick={() => { core.current?.elements().removeClass('muted highlighted').unselect(); }}><RotateCcw size={15} /></button></div>
    <div className="graph-watermark">CI-HGNN-RGD <span>/</span> {compact ? 'Graph preview' : 'Scholarly knowledge graph'}</div>
  </div>;
}
