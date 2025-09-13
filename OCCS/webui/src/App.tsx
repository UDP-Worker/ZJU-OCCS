import React, { useEffect, useMemo, useRef, useState } from 'react'
// Components below are wired inline for now; placeholders kept for future use
import { WaveformChart } from './components/WaveformChart'
import { LossChart } from './components/LossChart'
import { OptimizerControls } from './components/OptimizerControls'
import { StatusBar } from './components/StatusBar'
// Diagnostics chart for xi
import { XiChart } from './components/XiChart'
import { getBackends, createSession, getResponse, postVoltages, startOptimize, stopOptimize, getSessionStatus, getHistory, getVoltages, uploadTarget } from './api/client'
import { connectSessionStream, type StreamMessage } from './api/ws'

export default function App() {
  const [backends, setBackends] = useState<{ name: string; available: boolean }[]>([])
  const [backend, setBackend] = useState('mock')
  const [dac, setDac] = useState(3)
  const [bounds, setBounds] = useState<{ low: number; high: number }[]>([
    { low: -1, high: 1 },
    { low: -1, high: 1 },
    { low: -1, high: 1 },
  ])
  const [wlStart, setWlStart] = useState(1.55e-6)
  const [wlStop, setWlStop] = useState(1.56e-6)
  const [wlM, setWlM] = useState(96)

  const [sessionId, setSessionId] = useState<string | null>(null)
  const [status, setStatus] = useState<{ running: boolean; iter: number; best_loss: number | null }>({ running: false, iter: 0, best_loss: null })
  const [losses, setLosses] = useState<number[]>([])
  const [wave, setWave] = useState<{ lambda: number[]; signal: number[]; target: number[] }>({ lambda: [], signal: [], target: [] })
  const [currVolts, setCurrVolts] = useState<number[]>([])
  const [bestVolts, setBestVolts] = useState<number[] | null>(null)
  const [xi, setXi] = useState<number | null>(null)
  const [xis, setXis] = useState<number[]>([])
  const [targetPath, setTargetPath] = useState<string | null>(null)
  const [targetLabel, setTargetLabel] = useState<string | null>(null)

  const [voltsText, setVoltsText] = useState('0,0,0')
  const [nCalls, setNCalls] = useState(5)
  // 全局电压边界（快捷统一设置）
  const [globalLow, setGlobalLow] = useState(-1)
  const [globalHigh, setGlobalHigh] = useState(1)

  const wsRef = useRef<WebSocket | null>(null)
  const [wsReady, setWsReady] = useState(false)
  const [polling, setPolling] = useState(false)

  useEffect(() => {
    getBackends().then(setBackends).catch(() => setBackends([{ name: 'mock', available: true }]))
  }, [])

  useEffect(() => {
    // Clean up WS on unmount
    return () => {
      try { wsRef.current?.close() } catch {}
    }
  }, [])

  // Adjust bounds size when DAC changes
  useEffect(() => {
    setBounds((prev) => {
      const next = Array.from({ length: dac }, (_, i) => prev[i] ?? { low: globalLow, high: globalHigh })
      return next
    })
    setVoltsText(Array.from({ length: dac }).fill('0').join(','))
  }, [dac])

  // Fallback: if WS 未就绪但 session 存在，轮询状态与历史
  useEffect(() => {
    if (!sessionId || wsReady) return
    let stop = false
    let timer: any
    const tick = async () => {
      if (stop) return
      setPolling(true)
      try {
        const st = await getSessionStatus(sessionId)
        setStatus({ running: !!st.running, iter: st.iter ?? 0, best_loss: st.best_loss ?? null })
        if (st.x && Array.isArray(st.x)) setBestVolts(st.x as number[])
        // 拉取一次历史，用于更新 loss 曲线
        const hist = await getHistory(sessionId)
        const losses = (hist.history || []).map((h: any) => h.loss).filter((v: any) => typeof v === 'number')
        if (losses.length) setLosses(losses)
        // xi sequence from history diagnostics
        const xisHist = (hist.history || [])
          .map((h: any) => (h && h.diag ? h.diag.xi : undefined))
          .filter((v: any) => typeof v === 'number' && isFinite(v))
        if (xisHist.length) setXis(xisHist)
        // 最新 x 作为当前电压
        const last = (hist.history || []).slice(-1)[0]
        if (last && last.x && Array.isArray(last.x)) setCurrVolts(last.x as number[])
        if (last && last.diag && typeof last.diag.xi === 'number') setXi(last.diag.xi as number)
        // 可选：运行态时刷新波形与当前电压
        if (st.running) {
          try { setWave(await getResponse(sessionId)) } catch {}
          try { const vv = await getVoltages(sessionId); setCurrVolts(vv.volts) } catch {}
        }
      } catch {}
      timer = setTimeout(tick, 1000)
    }
    tick()
    return () => { stop = true; if (timer) clearTimeout(timer); setPolling(false) }
  }, [sessionId, wsReady])

  const canCreate = useMemo(() => !!backend && dac > 0 && wlM > 1 && wlStop > wlStart, [backend, dac, wlM, wlStart, wlStop])

  async function onCreateSession() {
    if (!canCreate) return
    const payload: any = { backend, dac_size: dac, wavelength: { start: wlStart, stop: wlStop, M: wlM }, bounds: bounds.map(b => [b.low, b.high]) }
    if (targetPath) payload.target_csv_path = targetPath
    const r = await createSession(payload)
    setSessionId(r.session_id)
    // reset default volt text to match dac size
    setVoltsText(Array.from({ length: dac }).fill('0').join(','))
    // connect WS
    const ws = connectSessionStream(r.session_id)
    wsRef.current = ws
    ws.onopen = () => setWsReady(true)
    ws.onmessage = (ev) => {
      setWsReady(true)
      const msg = JSON.parse(ev.data) as StreamMessage
      if (msg.type === 'status') {
        setStatus({ running: msg.running, iter: msg.iter, best_loss: (msg as any).best_loss ?? null })
        if ((msg as any).x && Array.isArray((msg as any).x)) setBestVolts((msg as any).x as number[])
      } else if (msg.type === 'progress') {
        setLosses((l) => [...l, msg.loss])
        if (msg.x && Array.isArray(msg.x)) setCurrVolts(msg.x)
        const bx: any = (msg as any).best_x
        if (bx && Array.isArray(bx)) setBestVolts(bx as number[])
        if (typeof (msg as any).xi === 'number') {
          const v = (msg as any).xi as number
          setXi(v)
          setXis((arr) => [...arr, v])
        }
      } else if (msg.type === 'waveform') {
        setWave({ lambda: msg.lambda, signal: msg.signal, target: msg.target })
      }
    }
    ws.onclose = () => { if (wsRef.current === ws) wsRef.current = null; setWsReady(false) }
    // fetch initial waveform snapshot
    try {
      const wf = await getResponse(r.session_id)
      setWave(wf)
      const vv = await getVoltages(r.session_id)
      setCurrVolts(vv.volts)
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
    try { const vv = await getVoltages(sessionId); setCurrVolts(vv.volts) } catch {}
  }

  async function onStartOptimize() {
    if (!sessionId) return
    setLosses([])
    setXis([])
    await startOptimize(sessionId, { n_calls: nCalls })
  }

  async function onStopOptimize() {
    if (!sessionId) return
    await stopOptimize(sessionId)
  }

  const connected = wsReady || polling

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
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, opacity: 0.8 }}>每通道电压范围（低/高，默认 -1..1）</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '6px 0' }}>
            <span style={{ fontSize: 12, opacity: 0.8 }}>统一设置：</span>
            <input type="number" step="any" value={globalLow} onChange={(e) => setGlobalLow(parseFloat(e.target.value))} style={{ width: 80 }} />
            <span>~</span>
            <input type="number" step="any" value={globalHigh} onChange={(e) => setGlobalHigh(parseFloat(e.target.value))} style={{ width: 80 }} />
            <button type="button" onClick={() => setBounds(Array.from({ length: dac }, () => ({ low: globalLow, high: globalHigh })))}>应用到所有通道</button>
            <button type="button" onClick={() => { if (bounds.length > 0) { setGlobalLow(bounds[0].low); setGlobalHigh(bounds[0].high) } }}>取 ch0</button>
          </div>
          {bounds.map((b, i) => (
            <div key={i} style={{ display: 'inline-flex', alignItems: 'center', marginRight: 8 }}>
              <span style={{ fontSize: 12, opacity: 0.8 }}>ch{i}:</span>
              <input type="number" step="any" value={b.low} onChange={(e) => {
                const v = parseFloat(e.target.value); setBounds(bs => bs.map((bb, j) => j === i ? { ...bb, low: v } : bb))
              }} style={{ width: 80, marginLeft: 4 }} />
              <span style={{ margin: '0 4px' }}>~</span>
              <input type="number" step="any" value={b.high} onChange={(e) => {
                const v = parseFloat(e.target.value); setBounds(bs => bs.map((bb, j) => j === i ? { ...bb, high: v } : bb))
              }} style={{ width: 80 }} />
            </div>
          ))}
        </div>
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, opacity: 0.8 }}>理想波形 CSV</div>
          <input type="file" accept=".csv" onChange={async (e) => {
            const f = e.currentTarget.files?.[0]
            if (!f) return
            try {
              const res = await uploadTarget(f)
              setTargetPath(res.path)
              setTargetLabel(f.name)
            } catch (e) {
              console.error('上传失败', e)
            }
          }} />
          {targetLabel ? <span style={{ marginLeft: 8, fontSize: 12, opacity: 0.8 }}>已上传：{targetLabel}</span> : null}
        </div>
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
        <div style={{ marginTop: 6, fontSize: 12, color: '#98a2b3' }}>
          当前电压：[{currVolts.map(v => Number.isFinite(v) ? v.toFixed(3) : String(v)).join(', ')}]
        </div>
        <div style={{ marginTop: 4, fontSize: 12, color: '#98a2b3' }}>
          最优电压：[{(bestVolts ?? []).map(v => Number.isFinite(v) ? v.toFixed(3) : String(v)).join(', ')}]
        </div>
      </fieldset>

      <fieldset>
        <legend>优化控制</legend>
        <label>
          迭代次数
          <input type="number" min={1} value={nCalls} onChange={(e) => setNCalls(parseInt(e.target.value || '1'))} />
        </label>
        <button type="button" disabled={!sessionId} onClick={onStartOptimize}>开始优化</button>
        <button type="button" disabled={!sessionId} onClick={onStopOptimize}>停止</button>
        {sessionId ? (
          <a style={{ marginLeft: 8 }} href={`/api/session/${sessionId}/history.csv`} target="_blank" rel="noreferrer">
            下载历史 CSV
          </a>
        ) : null}
        <div style={{ marginTop: 8, fontSize: 12, color: '#888' }}>
          状态：{status.running ? '运行中' : '空闲'}，迭代：{status.iter}，best_loss：{status.best_loss ?? '—'}，loss 点数：{losses.length}
        </div>
      </fieldset>

      <fieldset>
        <legend>诊断</legend>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
          <div style={{ flex: 1 }}>
            <XiChart xis={xis} />
          </div>
          <div style={{ minWidth: 160 }}>
            <div style={{ fontSize: 12, color: '#98a2b3' }}>当前 xi</div>
            <div style={{ fontSize: 18 }}>{xi != null && isFinite(xi) ? xi.toFixed(4) : '—'}</div>
          </div>
        </div>
      </fieldset>

      <StatusBar connected={!!connected} sessionId={sessionId} />
    </div>
  )
}
