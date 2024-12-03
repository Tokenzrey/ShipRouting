declare module 'graph-data-structure' {
  interface GraphNodeData {
    category: number;
    details: string;
  }

  export default function Graph(): {
    addNode: (key: string, data?: GraphNodeData) => void;
    removeNode: (key: string) => void;
    hasNode: (key: string) => boolean;
    adjacent: (key: string) => string[];
    addEdge: (from: string, to: string) => void;
    removeEdge: (from: string, to: string) => void;
    serialize: () => object;
    getNodeData: (key: string) => GraphNodeData | undefined;
    setNodeData: (key: string, data: GraphNodeData) => void;
  };
}
