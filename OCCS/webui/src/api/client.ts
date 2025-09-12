export type BackendInfo = { name: string; available: boolean }

const BASE = ''

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return (await r.json()) as T
}

export async function getBackends() {
  return http<BackendInfo[]>('/api/backends')
}

export async function createSession(payload: any) {
  return http<{ session_id: string }>('/api/session', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function getSessionStatus(id: string) {
  return http<any>(`/api/session/${id}/status`)
}

export async function deleteSession(id: string) {
  return http<{ ok: boolean }>(`/api/session/${id}`, { method: 'DELETE' })
}

export async function postVoltages(id: string, volts: number[]) {
  return http<{ ok: boolean }>(`/api/session/${id}/voltages`, {
    method: 'POST',
    body: JSON.stringify({ volts }),
  })
}

export async function getResponse(id: string) {
  return http<{ lambda: number[]; signal: number[]; target: number[] }>(
    `/api/session/${id}/response`
  )
}

export async function getVoltages(id: string) {
  return http<{ volts: number[] }>(`/api/session/${id}/voltages`)
}

export async function startOptimize(id: string, payload: any) {
  return http<{ ok: boolean }>(`/api/session/${id}/optimize/start`, {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function stopOptimize(id: string) {
  return http<{ ok: boolean }>(`/api/session/${id}/optimize/stop`, { method: 'POST' })
}

export async function getHistory(id: string) {
  return http<{ history: any[]; best_loss: number | null; best_x: number[] | null }>(
    `/api/session/${id}/history`
  )
}
