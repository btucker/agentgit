/**
 * ReframeViewer - React component for visualizing mental models
 *
 * Uses react-force-graph for interactive force-directed graph visualization.
 * Loads model data from .agentgit/mental_model.json
 *
 * Install dependencies:
 *   npm install react-force-graph-2d
 *
 * Usage:
 *   <ReframeViewer modelPath="/path/to/.agentgit/mental_model.json" />
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

interface Node {
  id: string;
  name: string;
  group: string;
  color?: string;
  val?: number;
  description?: string;
  files?: string[];
}

interface Link {
  source: string;
  target: string;
  label?: string;
  style?: string;
  description?: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface MentalModelData {
  elements: Record<string, any>;
  relations: any[];
  version: number;
  ai_summary: string;
}

// Color palette for groups
const GROUP_COLORS: Record<string, string> = {
  box: "#90caf9",
  rounded: "#a5d6a7",
  circle: "#ffcc80",
  diamond: "#ce93d8",
  cylinder: "#80deea",
  hexagon: "#f48fb1",
  stadium: "#bcaaa4",
  default: "#e0e0e0",
};

function transformToForceGraph(model: MentalModelData): GraphData {
  const nodes: Node[] = Object.values(model.elements).map((elem: any) => ({
    id: elem.id,
    name: elem.label,
    group: elem.properties?.group || elem.shape || "default",
    color: elem.color,
    val: elem.properties?.size || 1,
    description: elem.reasoning,
  }));

  const links: Link[] = model.relations.map((rel: any) => ({
    source: rel.source_id,
    target: rel.target_id,
    label: rel.label,
    style: rel.style,
    description: rel.reasoning,
  }));

  return { nodes, links };
}

interface ReframeViewerProps {
  modelPath?: string;
  modelData?: MentalModelData;
  width?: number;
  height?: number;
  onNodeClick?: (node: Node) => void;
  onNodeHover?: (node: Node | null) => void;
}

export function ReframeViewer({
  modelPath,
  modelData,
  width = 800,
  height = 600,
  onNodeClick,
  onNodeHover,
}: ReframeViewerProps) {
  const graphRef = useRef<any>();
  const [data, setData] = useState<GraphData>({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [hoveredNode, setHoveredNode] = useState<Node | null>(null);

  // Load model from file or use provided data
  useEffect(() => {
    if (modelData) {
      setData(transformToForceGraph(modelData));
    } else if (modelPath) {
      fetch(modelPath)
        .then((res) => res.json())
        .then((model) => setData(transformToForceGraph(model)))
        .catch((err) => console.error("Failed to load model:", err));
    }
  }, [modelPath, modelData]);

  // Node color based on group
  const nodeColor = useCallback(
    (node: Node) => {
      if (node.color) return node.color;
      if (selectedNode?.id === node.id) return "#ff5722";
      if (hoveredNode?.id === node.id) return "#ffc107";
      return GROUP_COLORS[node.group] || GROUP_COLORS.default;
    },
    [selectedNode, hoveredNode]
  );

  // Link styling
  const linkColor = useCallback((link: Link) => {
    return link.style === "dashed" ? "#999" : "#666";
  }, []);

  const linkWidth = useCallback((link: Link) => {
    return link.style === "thick" ? 3 : 1;
  }, []);

  // Node click handler
  const handleNodeClick = useCallback(
    (node: Node) => {
      setSelectedNode(node);
      onNodeClick?.(node);

      // Center on clicked node
      graphRef.current?.centerAt(node.x, node.y, 1000);
      graphRef.current?.zoom(2, 1000);
    },
    [onNodeClick]
  );

  // Node hover handler
  const handleNodeHover = useCallback(
    (node: Node | null) => {
      setHoveredNode(node);
      onNodeHover?.(node);
    },
    [onNodeHover]
  );

  // Custom node rendering
  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const label = node.name;
      const fontSize = 12 / globalScale;
      ctx.font = `${fontSize}px Sans-Serif`;

      // Node circle
      const size = Math.sqrt(node.val || 1) * 5;
      ctx.beginPath();
      ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
      ctx.fillStyle = nodeColor(node);
      ctx.fill();

      // Border for selected/hovered
      if (selectedNode?.id === node.id || hoveredNode?.id === node.id) {
        ctx.strokeStyle = selectedNode?.id === node.id ? "#ff5722" : "#ffc107";
        ctx.lineWidth = 2 / globalScale;
        ctx.stroke();
      }

      // Label
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#333";
      ctx.fillText(label, node.x, node.y + size + fontSize);
    },
    [nodeColor, selectedNode, hoveredNode]
  );

  // Link label rendering
  const linkCanvasObjectMode = () => "after";
  const linkCanvasObject = useCallback(
    (link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      if (!link.label) return;

      const fontSize = 10 / globalScale;
      ctx.font = `${fontSize}px Sans-Serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#666";

      // Position at midpoint
      const midX = (link.source.x + link.target.x) / 2;
      const midY = (link.source.y + link.target.y) / 2;
      ctx.fillText(link.label, midX, midY);
    },
    []
  );

  return (
    <div style={{ position: "relative" }}>
      <ForceGraph2D
        ref={graphRef}
        graphData={data}
        width={width}
        height={height}
        nodeColor={nodeColor}
        nodeVal={(node: Node) => node.val || 1}
        nodeCanvasObject={nodeCanvasObject}
        nodeCanvasObjectMode={() => "replace"}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkCanvasObject={linkCanvasObject}
        linkCanvasObjectMode={linkCanvasObjectMode}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        cooldownTicks={100}
        onEngineStop={() => graphRef.current?.zoomToFit(400)}
      />

      {/* Info panel for selected node */}
      {selectedNode && (
        <div
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            background: "white",
            padding: 16,
            borderRadius: 8,
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            maxWidth: 300,
          }}
        >
          <h3 style={{ margin: "0 0 8px" }}>{selectedNode.name}</h3>
          {selectedNode.description && (
            <p style={{ margin: "0 0 8px", color: "#666" }}>
              {selectedNode.description}
            </p>
          )}
          {selectedNode.files && selectedNode.files.length > 0 && (
            <div>
              <strong>Files:</strong>
              <ul style={{ margin: "4px 0", paddingLeft: 20 }}>
                {selectedNode.files.map((f) => (
                  <li key={f} style={{ fontSize: 12 }}>
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <button
            onClick={() => setSelectedNode(null)}
            style={{
              marginTop: 8,
              padding: "4px 12px",
              cursor: "pointer",
            }}
          >
            Close
          </button>
        </div>
      )}

      {/* Hover tooltip */}
      {hoveredNode && hoveredNode !== selectedNode && (
        <div
          style={{
            position: "absolute",
            bottom: 10,
            left: 10,
            background: "rgba(0,0,0,0.8)",
            color: "white",
            padding: "4px 8px",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          {hoveredNode.name}
          {hoveredNode.description && `: ${hoveredNode.description}`}
        </div>
      )}
    </div>
  );
}

export default ReframeViewer;
