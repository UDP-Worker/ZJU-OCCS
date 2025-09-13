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

  // 手动电压输入（每通道一个输入框）
  const [voltsArr, setVoltsArr] = useState<number[]>([0, 0, 0])
  const [nCalls, setNCalls] = useState(5)
  // 可选：优化的随机种子（留空表示不指定）
  const [randSeed, setRandSeed] = useState<string>('')
  // 全局电压边界（快捷统一设置）
  const [globalLow, setGlobalLow] = useState(-1)
  const [globalHigh, setGlobalHigh] = useState(1)

  const wsRef = useRef<WebSocket | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const [wsReady, setWsReady] = useState(false)
  const [polling, setPolling] = useState(false)
  // 主题：auto | light | dark
  const [theme, setTheme] = useState<'auto' | 'light' | 'dark'>('auto')
  useEffect(() => {
    try {
      const saved = localStorage.getItem('occs-theme')
      if (saved === 'light' || saved === 'dark' || saved === 'auto') {
        setTheme(saved)
      }
    } catch {}
  }, [])

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'auto') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    try { localStorage.setItem('occs-theme', theme) } catch {}
  }, [theme])

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
    setVoltsArr(prev => Array.from({ length: dac }, (_, i) => (prev as number[])[i] ?? 0))
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
    // reset manual voltages to zeros matching dac size
    setVoltsArr(Array.from({ length: dac }).map(() => 0))
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
    if (!sessionId || status.running) return
    const arr = Array.from({ length: dac }, (_, i) => Number(voltsArr[i] ?? 0))
    await postVoltages(sessionId, arr as number[])
    const wf = await getResponse(sessionId)
    setWave(wf)
    try { const vv = await getVoltages(sessionId); setCurrVolts(vv.volts) } catch {}
  }

  async function onApplyBestVoltages() {
    if (!sessionId || status.running) return
    if (!bestVolts || !bestVolts.length) return
    await postVoltages(sessionId, bestVolts)
    try {
      const wf = await getResponse(sessionId)
      setWave(wf)
      const vv = await getVoltages(sessionId)
      setCurrVolts(vv.volts)
      setVoltsArr(bestVolts.slice())
    } catch {}
  }

  async function onCopyBestVoltages() {
    const arr = bestVolts ?? []
    if (!arr.length) return
    const text = arr.join(',')
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        // Fallback: use a temporary textarea
        const ta = document.createElement('textarea')
        ta.value = text
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        document.body.removeChild(ta)
      }
    } catch (e) {
      // noop on failure
      console.error('复制失败', e)
    }
  }

  async function onStartOptimize() {
    if (!sessionId) return
    setLosses([])
    setXis([])
    const payload: any = { n_calls: nCalls }
    const s = String(randSeed ?? '').trim()
    if (s !== '' && !Number.isNaN(Number(s))) {
      try {
        payload.random_state = parseInt(s)
      } catch {}
    }
    await startOptimize(sessionId, payload)
  }

  async function onStopOptimize() {
    if (!sessionId) return
    await stopOptimize(sessionId)
  }

  const connected = wsReady || polling

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <h1>OCCS Web UI</h1>
        <div className="row">
          <button className={`btn ${theme==='auto' ? 'btn-primary' : ''}`} onClick={() => setTheme('auto')}>系统</button>
          <button className={`btn ${theme==='dark' ? 'btn-primary' : ''}`} onClick={() => setTheme('dark')}>深色</button>
          <button className={`btn ${theme==='light' ? 'btn-primary' : ''}`} onClick={() => setTheme('light')}>浅色</button>
        </div>
      </div>

      {/* 创建会话 */}
      <div className="card">
        <div className="card-title">创建会话</div>
        <div className="row">
          <span className="label">后端</span>
          <select className="select" value={backend} onChange={(e) => setBackend(e.target.value)}>
            {backends.map((b) => (
              <option key={b.name} value={b.name} disabled={!b.available}>
                {b.name} {b.available ? '' : '(不可用)'}
              </option>
            ))}
          </select>
          <span className="label">DAC 通道</span>
          <input className="input" type="number" min={1} value={dac} onChange={(e) => setDac(parseInt(e.target.value || '1'))} />
          <span className="label">波长 start (m)</span>
          <input className="input" type="number" step="any" value={wlStart} onChange={(e) => setWlStart(parseFloat(e.target.value))} />
          <span className="label">stop</span>
          <input className="input" type="number" step="any" value={wlStop} onChange={(e) => setWlStop(parseFloat(e.target.value))} />
          <span className="label">M</span>
          <input className="input" type="number" min={2} value={wlM} onChange={(e) => setWlM(parseInt(e.target.value || '2'))} />
        </div>
        <div className="spacer"></div>
        <div className="muted">每通道电压范围（低/高，默认 -1..1）</div>
        <div className="row" style={{ margin: '6px 0' }}>
          <span className="label">统一设置</span>
          <input className="input" type="number" step="any" value={globalLow} onChange={(e) => setGlobalLow(parseFloat(e.target.value))} style={{ width: 100 }} />
          <span>~</span>
          <input className="input" type="number" step="any" value={globalHigh} onChange={(e) => setGlobalHigh(parseFloat(e.target.value))} style={{ width: 100 }} />
          <button className="btn" type="button" onClick={() => setBounds(Array.from({ length: dac }, () => ({ low: globalLow, high: globalHigh })))}>应用到所有通道</button>
          <button className="btn" type="button" onClick={() => { if (bounds.length > 0) { setGlobalLow(bounds[0].low); setGlobalHigh(bounds[0].high) } }}>取 ch0</button>
        </div>
        <div className="row" style={{ flexWrap: 'wrap' }}>
          {bounds.map((b, i) => (
            <div key={i} style={{ display: 'inline-flex', alignItems: 'center', marginRight: 8 }}>
              <span className="label">ch{i}</span>
              <input className="input" type="number" step="any" value={b.low} onChange={(e) => {
                const v = parseFloat(e.target.value); setBounds(bs => bs.map((bb, j) => j === i ? { ...bb, low: v } : bb))
              }} style={{ width: 100 }} />
              <span style={{ margin: '0 4px' }}>~</span>
              <input className="input" type="number" step="any" value={b.high} onChange={(e) => {
                const v = parseFloat(e.target.value); setBounds(bs => bs.map((bb, j) => j === i ? { ...bb, high: v } : bb))
              }} style={{ width: 100 }} />
            </div>
          ))}
        </div>
        <div className="spacer"></div>
        <div className="muted">理想波形 CSV</div>
        <div className="row">
          <input ref={fileRef} style={{ display: 'none' }} type="file" accept=".csv" onChange={async (e) => {
            const f = e.currentTarget.files?.[0]
            if (!f) return
            try {
              const res = await uploadTarget(f)
              setTargetPath(res.path)
              setTargetLabel(f.name)
            } catch (e) {
              console.error('上传失败', e)
            } finally {
              e.currentTarget.value = '' // reset for same-file reselect
            }
          }} />
          <button className="btn" type="button" onClick={() => fileRef.current?.click()}>选择文件</button>
          {targetLabel ? <span className="muted">已上传：{targetLabel}</span> : <span className="muted">未选择文件</span>}
        </div>
        <div className="spacer"></div>
        <div className="row">
          <button className="btn btn-primary" type="button" disabled={!!sessionId || !canCreate} onClick={onCreateSession}>创建</button>
          {sessionId ? <span className="muted">会话: {sessionId.slice(0, 8)}…</span> : null}
        </div>
      </div>

      <div className="grid grid-2" style={{ marginTop: 16 }}>
        <div className="card stretch">
          <div className="card-title">波形</div>
          <WaveformChart data={wave} themeKey={theme} />
        </div>
        <div className="card stretch">
          <div className="card-title">Loss 曲线</div>
          <LossChart losses={losses} themeKey={theme} />
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">手动电压</div>
        <div className="muted" style={{ marginBottom: 6 }}>每通道一个输入，单位：V</div>
        <div className="row" style={{ flexWrap: 'wrap' }}>
          {Array.from({ length: dac }).map((_, i) => (
            <label key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span className="label">ch{i}</span>
              <input
                className="input" type="number"
                step="any"
                value={Number.isFinite(voltsArr[i]) ? voltsArr[i] : 0}
                onChange={(e) => {
                  const v = e.target.value
                  const num = v === '' ? 0 : parseFloat(v)
                  setVoltsArr((arr) => arr.map((vv, j) => (j === i ? (Number.isNaN(num) ? 0 : num) : vv)))
                }}
                disabled={status.running || !sessionId}
                style={{ width: 100 }}
              />
            </label>
          ))}
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button className="btn btn-primary" type="button" disabled={!sessionId || status.running} onClick={onApplyVoltages}>应用并刷新波形</button>
          <button className="btn" type="button" disabled={!sessionId || status.running || !bestVolts || bestVolts.length === 0} onClick={onApplyBestVoltages}>应用最优电压</button>
          <button className="btn" type="button" disabled={!bestVolts || bestVolts.length === 0} onClick={onCopyBestVoltages}>复制最优电压</button>
        </div>
        {status.running ? (
          <div className="muted" style={{ marginTop: 6, color: '#d97706' }}>
            优化进行中，已暂时禁止手动下发电压。
          </div>
        ) : null}
        <div className="muted" style={{ marginTop: 6 }}>
          当前电压：[{currVolts.map(v => Number.isFinite(v) ? v.toFixed(3) : String(v)).join(', ')}]
        </div>
        <div className="muted" style={{ marginTop: 4 }}>
          最优电压：[{(bestVolts ?? []).map(v => Number.isFinite(v) ? v.toFixed(3) : String(v)).join(', ')}]
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">优化控制</div>
        <div className="row">
          <span className="label">迭代次数</span>
          <input className="input" type="number" min={1} value={nCalls} onChange={(e) => setNCalls(parseInt(e.target.value || '1'))} />
          <span className="label">随机种子</span>
          <input className="input" type="number" placeholder="可选，如 42" value={randSeed} onChange={(e) => setRandSeed(e.target.value)} style={{ width: 140 }} />
          <button className="btn btn-primary" type="button" disabled={!sessionId} onClick={onStartOptimize}>开始优化</button>
          <button className="btn" type="button" disabled={!sessionId} onClick={onStopOptimize}>停止</button>
          {sessionId ? (
            <a className="btn btn-ghost" href={`/api/session/${sessionId}/history.csv`} target="_blank" rel="noreferrer">下载历史 CSV</a>
          ) : null}
        </div>
        <div className="muted" style={{ marginTop: 8 }}>
          状态：{status.running ? '运行中' : '空闲'}，迭代：{status.iter}，best_loss：{status.best_loss ?? '—'}，loss 点数：{losses.length}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-title">诊断</div>
        <div className="row" style={{ alignItems: 'flex-start' }}>
          <div style={{ flex: 1 }}>
            <XiChart xis={xis} themeKey={theme} />
          </div>
          <div style={{ minWidth: 160 }}>
            <div className="muted">当前 ξ</div>
            <div style={{ fontSize: 18 }}>{xi != null && isFinite(xi) ? xi.toFixed(4) : '—'}</div>
          </div>
        </div>
      </div>

      <StatusBar connected={!!connected} sessionId={sessionId} />
    </div>
  )
}
