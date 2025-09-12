export type StreamMessage =
  | { type: 'status'; running: boolean; iter: number; best_loss?: number | null }
  | { type: 'waveform'; lambda: number[]; signal: number[]; target: number[] }
  | { type: 'progress'; iter: number; loss: number; running_min: number; x: number[]; xi?: number | null; kappa?: number | null; gp_max_std?: number | null }
  | { type: 'done'; best_loss: number | null }
  | { type: 'error'; message: string }

export function connectSessionStream(sessionId: string): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/api/session/${sessionId}/stream`)
  return ws
}

