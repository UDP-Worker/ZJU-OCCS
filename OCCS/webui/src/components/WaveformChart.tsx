import React, { useEffect, useRef } from 'react'

export function WaveformChart({ data }: { data: { lambda: number[]; signal: number[]; target: number[] } }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const cvs = canvasRef.current
    if (!cvs) return
    const ctx = cvs.getContext('2d')!
    const w = (cvs.width = cvs.clientWidth)
    const h = (cvs.height = cvs.clientHeight)
    ctx.clearRect(0, 0, w, h)
    ctx.fillStyle = '#11151c'
    ctx.fillRect(0, 0, w, h)
    const { lambda, signal, target } = data
    if (!lambda?.length || !signal?.length) {
      ctx.fillStyle = '#98a2b3'
      ctx.fillText('暂无波形数据', 10, 20)
      return
    }
    const N = Math.min(lambda.length, signal.length)
    const sig = signal.slice(0, N)
    const tgt = target?.length ? target.slice(0, Math.min(N, target.length)) : []
    const minv = Math.min(...sig, ...(tgt.length ? tgt : [Infinity]))
    const maxv = Math.max(...sig, ...(tgt.length ? tgt : [-Infinity]))
    const yScale = (v: number) => {
      const denom = maxv - minv || 1
      return h - ((v - minv) / denom) * (h - 20) - 10
    }
    const xScale = (i: number) => (i / (N - 1 || 1)) * (w - 20) + 10
    // grid
    ctx.strokeStyle = '#2a2f3a'
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let i = 0; i <= 4; i++) {
      const yy = (i / 4) * (h - 20) + 10
      ctx.moveTo(10, yy)
      ctx.lineTo(w - 10, yy)
    }
    ctx.stroke()
    // target
    if (tgt.length) {
      ctx.strokeStyle = '#e57373'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(xScale(0), yScale(tgt[0]))
      for (let i = 1; i < tgt.length; i++) ctx.lineTo(xScale(i), yScale(tgt[i]))
      ctx.stroke()
    }
    // signal
    ctx.strokeStyle = '#5b9cf6'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(xScale(0), yScale(sig[0]))
    for (let i = 1; i < sig.length; i++) ctx.lineTo(xScale(i), yScale(sig[i]))
    ctx.stroke()

    // legend
    const lx = 16, ly = 16, ll = 28
    // signal legend
    ctx.strokeStyle = '#5b9cf6'; ctx.lineWidth = 2
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + ll, ly); ctx.stroke()
    ctx.fillStyle = '#e6e9ef'; ctx.fillText('响应', lx + ll + 6, ly + 4)
    // target legend
    ctx.strokeStyle = '#e57373'; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(lx, ly + 16); ctx.lineTo(lx + ll, ly + 16); ctx.stroke()
    ctx.fillStyle = '#e6e9ef'; ctx.fillText('目标', lx + ll + 6, ly + 20)
  }, [data])
  return (
    <div>
      <h3>波形</h3>
      <div style={{ height: 240, border: '1px solid #2a2f3a' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>
    </div>
  )
}
