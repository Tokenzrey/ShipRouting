// src/lib/WebSocketClient.ts

import { useRouteStore, PathPoint, BlockedEdge } from './GlobalState/state';

interface Expansion {
  expanded_node: number;
  current_distance: number;
  longitude: number;
  latitude: number;
}

interface PartialPathMessage {
  type: 'partial_path';
  path: PathPoint[];
  current_best_distance: number;
}

interface ExpansionMessage {
  type: 'expansion';
  batch: Expansion[];
}

interface FinalMessage {
  type: 'final';
  path: PathPoint[];
  distance: number;
  blocked_edges?: BlockedEdge[];
}

type IncomingMessage = PartialPathMessage | ExpansionMessage | FinalMessage;

export class WebSocketHandler {
  private socket: WebSocket | null = null;
  private url: string;
  private reconnectInterval = 5000;
  private sendQueue: any[] = [];
  private isConnected = false;

  // Tambahkan event handler callback
  public onClose: (() => void) | null = null;
  public onError: (() => void) | null = null;
  public onFinal: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    this.connect();
  }

  private connect() {
    this.socket = new WebSocket(this.url);

    this.socket.onopen = () => {
      console.info('WebSocket connection established.');
      this.isConnected = true;

      // Kirim semua pesan yang tertunda
      while (this.sendQueue.length > 0) {
        const message = this.sendQueue.shift();
        this.sendMessage(message);
      }
    };

    this.socket.onmessage = (event) => {
      try {
        const message: IncomingMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
        if (this.onError) this.onError(); // Panggil onError jika ada error
      }
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (this.onError) this.onError(); // Panggil onError jika terjadi error
    };

    this.socket.onclose = (event) => {
      console.warn(
        `WebSocket closed: Code ${event.code}, Reason: ${event.reason}`,
      );
      this.isConnected = false;
      if (this.onClose) this.onClose(); // Panggil onClose jika WebSocket ditutup

      // Reconnect setelah delay
      setTimeout(() => {
        console.info('Reconnecting WebSocket...');
        this.connect();
      }, this.reconnectInterval);
    };
  }

  private handleMessage(message: IncomingMessage) {
    const store = useRouteStore.getState();

    if ('type' in message) {
      switch (message.type) {
        case 'expansion':
          store.addExpansions(
            message.batch.map((expansion) => ({
              ...expansion,
              coordinates: [expansion.longitude, expansion.latitude],
            })),
          );
          break;
        case 'partial_path':
          store.setPartialPath(message.path);
          break;
        case 'final':
          store.setFinalPath(message.path, message.distance);
          if (message.blocked_edges) {
            store.setBlockedEdges(message.blocked_edges);
          }
          if (this.onFinal) this.onFinal(); // Panggil onFinal ketika jalur final diterima
          break;
        default:
          console.warn('Unknown message type:', message);
      }
    } else {
      console.warn('Received message with unknown structure:', message);
    }
  }

  public sendMessage(message: any) {
    if (this.isConnected && this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not open. Queueing message:', message);
      this.sendQueue.push(message);
    }
  }

  public close() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}
