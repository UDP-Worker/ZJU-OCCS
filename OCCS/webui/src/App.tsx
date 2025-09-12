import React, { useEffect, useMemo, useRef, useState } from 'react'
// Components below are wired inline for now; placeholders kept for future use
import { WaveformChart } from './components/WaveformChart'
import { LossChart } from './components/LossChart'
import { OptimizerControls } from './components/OptimizerControls'
import { StatusBar } from './components/StatusBar'
import { getBackends, createSession, getResponse, postVoltages, startOptimize, stopOptimize } from './api/client'
import { connectSessionStream, type StreamMessage } from './api/ws'

export default function App() {
  const [backends, setBackends] = useState<{ name: string; available: boolean }[]>([])
  const [backend, setBackend] = useState('mock')
  const [dac, setDac] = useState(3)
  const [wlStart, setWlStart] = useState(1.55e-6)
  const [wlStop, setWlStop] = useState(1.56e-6)
  const [wlM, setWlM] = useState(96)

  const [sessionId, setSessionId] = useState<string | null>(null)
  const [status, setStatus] = useState<{ running: boolean; iter: number; best_loss: number | null }>({ running: false, iter: 0, best_loss: null })
  const [losses, setLosses] = useState<number[]>([])
  const [wave, setWave] = useState<{ lambda: number[]; signal: number[]; target: number[] }>({ lambda: [], signal: [], target: [] })

  const [voltsText, setVoltsText] = useState('0,0,0')
  const [nCalls, setNCalls] = useState(5)

  const wsRef = useRef<WebSocket | null>(null)
  const [wsReady, setWsReady] = useState(false)

  useEffect(() => {
    getBackends().then(setBackends).catch(() => setBackends([{ name: 'mock', available: true }]))
  }, [])

  useEffect(() => {
    // Clean up WS on unmount
    return () => {
      try { wsRef.current?.close() } catch {}
    }
  }, [])

  const canCreate = useMemo(() => !!backend && dac > 0 && wlM > 1 && wlStop > wlStart, [backend, dac, wlM, wlStart, wlStop])

  async function onCreateSession() {
    if (!canCreate) return
    const payload = { backend, dac_size: dac, wavelength: { start: wlStart, stop: wlStop, M: wlM } }
    const r = await createSession(payload)
    setSessionId(r.session_id)
    // reset default volt text to match dac size
    setVoltsText(Array.from({ length: dac }).fill('0').join(','))
    // connect WS
    const ws = connectSessionStream(r.session_id)
    wsRef.current = ws
    ws.onopen = () => setWsReady(true)
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as StreamMessage
      if (msg.type === 'status') {
        setStatus({ running: msg.running, iter: msg.iter, best_loss: (msg as any).best_loss ?? null })
      } else if (msg.type === 'progress') {
        setLosses((l) => [...l, msg.loss])
      } else if (msg.type === 'waveform') {
        setWave({ lambda: msg.lambda, signal: msg.signal, target: msg.target })
      }
    }
    ws.onclose = () => { if (wsRef.current === ws) wsRef.current = null; setWsReady(false) }
    // fetch initial waveform snapshot
    try {
      const wf = await getResponse(r.session_id)
      setWave(wf)
    } catch {
      // ignore
    }
  }

  async function onApplyVoltages() {
    if (!sessionId) return
    const arr = voltsText.split(/[ ,]+/).filter(Boolean).map(parseFloat)
    await postVoltages(sessionId, arr)
    const wf = await getResponse(sessionId)
    setWave(wf)
  }

  async function onStartOptimize() {
    if (!sessionId) return
    setLosses([])
    await startOptimize(sessionId, { n_calls: nCalls })
  }

  async function onStopOptimize() {
    if (!sessionId) return
    await stopOptimize(sessionId)
  }

  const connected = wsReady

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16 }}>
      <h1>OCCS Web UI</h1>

      {/* Quick controls (wired) */}
      <fieldset>
        <legend>创建会话</legend>
        <label>
          后端：
          <select value={backend} onChange={(e) => setBackend(e.target.value)}>
            {backends.map((b) => (
              <option key={b.name} value={b.name} disabled={!b.available}>
                {b.name} {b.available ? '' : '(不可用)'}
              </option>
            ))}
          </select>
        </label>
        <label>
          DAC 通道
          <input type="number" min={1} value={dac} onChange={(e) => setDac(parseInt(e.target.value || '1'))} />
        </label>
        <label>
          波长 start (m)
          <input type="number" step="any" value={wlStart} onChange={(e) => setWlStart(parseFloat(e.target.value))} />
        </label>
        <label>
          stop
          <input type="number" step="any" value={wlStop} onChange={(e) => setWlStop(parseFloat(e.target.value))} />
        </label>
        <label>
          M
          <input type="number" min={2} value={wlM} onChange={(e) => setWlM(parseInt(e.target.value || '2'))} />
        </label>
        <button type="button" disabled={!!sessionId || !canCreate} onClick={onCreateSession}>创建</button>
        {sessionId ? <span style={{ marginLeft: 8 }}>会话: {sessionId.slice(0, 8)}…</span> : null}
      </fieldset>

      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1 }}>
          <WaveformChart data={wave} />
        </div>
        <div style={{ flex: 1 }}>
          <LossChart losses={losses} />
        </div>
      </div>

      <fieldset>
        <legend>手动电压</legend>
        <input style={{ minWidth: 240 }} type="text" value={voltsText} onChange={(e) => setVoltsText(e.target.value)} />
        <button type="button" disabled={!sessionId} onClick={onApplyVoltages}>应用并刷新波形</button>
      </fieldset>

      <fieldset>
        <legend>优化控制</legend>
        <label>
          迭代次数
          <input type="number" min={1} value={nCalls} onChange={(e) => setNCalls(parseInt(e.target.value || '1'))} />
        </label>
        <button type="button" disabled={!sessionId} onClick={onStartOptimize}>开始优化</button>
        <button type="button" disabled={!sessionId} onClick={onStopOptimize}>停止</button>
        <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
          状态：{status.running ? '运行中' : '空闲'}，迭代：{status.iter}，best_loss：{status.best_loss ?? '—'}，loss 点数：{losses.length}
        </div>
      </fieldset>

      <StatusBar connected={!!connected} sessionId={sessionId} />
    </div>
  )
}
