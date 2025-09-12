import React, { useEffect, useRef } from 'react'

export function LossChart({ losses }: { losses: number[] }) {
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
    if (!losses?.length) {
      ctx.fillStyle = '#98a2b3'
      ctx.fillText('暂无 loss 数据', 10, 20)
      return
    }
    const minv = Math.min(...losses)
    const maxv = Math.max(...losses)
    const yScale = (v: number) => {
      const denom = maxv - minv || 1
      return h - ((v - minv) / denom) * (h - 20) - 10
    }
    const xScale = (i: number) => (i / (losses.length - 1 || 1)) * (w - 20) + 10
    // grid
    ctx.strokeStyle = '#2a2f3a'
    ctx.beginPath()
    for (let i = 0; i <= 4; i++) {
      const yy = (i / 4) * (h - 20) + 10
      ctx.moveTo(10, yy)
      ctx.lineTo(w - 10, yy)
    }
    ctx.stroke()
    // line
    ctx.strokeStyle = '#7bd88f'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(xScale(0), yScale(losses[0]))
    for (let i = 1; i < losses.length; i++) ctx.lineTo(xScale(i), yScale(losses[i]))
    ctx.stroke()
  }, [losses])
  return (
    <div>
      <h3>Loss 曲线</h3>
      <div style={{ height: 240, border: '1px solid #2a2f3a' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>
    </div>
  )
}
