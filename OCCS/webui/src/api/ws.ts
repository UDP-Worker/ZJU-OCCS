export type StreamMessage =
  | { type: 'status'; running: boolean; iter: number; best_loss?: number | null; x?: number[] | null }
  | { type: 'waveform'; lambda: number[]; signal: number[]; target: number[] }
  | { type: 'progress'; iter: number; loss: number; running_min: number; x: number[]; best_x?: number[] | null; xi?: number | null; kappa?: number | null; gp_max_std?: number | null }
  | { type: 'done'; best_loss: number | null }
  | { type: 'error'; message: string }

export function connectSessionStream(sessionId: string): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  // In dev (Vite on 5173), directly target the backend to avoid proxy WS issues
  const port = location.port
  const host = port === '5173' ? '127.0.0.1:8000' : location.host
  const url = `${proto}://${host}/api/session/${sessionId}/stream`
  return new WebSocket(url)
}
