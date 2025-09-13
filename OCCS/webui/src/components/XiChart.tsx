import React, { useEffect, useRef } from 'react'

export function XiChart({ xis, themeKey }: { xis: number[]; themeKey?: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    const cvs = canvasRef.current
    if (!cvs) return
    const ctx = cvs.getContext('2d')!
    const w = (cvs.width = cvs.clientWidth)
    const h = (cvs.height = cvs.clientHeight)
    ctx.clearRect(0, 0, w, h)
    const styles = getComputedStyle(document.documentElement)
    const card = styles.getPropertyValue('--card').trim() || '#11151c'
    const border = styles.getPropertyValue('--border').trim() || '#2a2f3a'
    const muted = styles.getPropertyValue('--muted').trim() || '#98a2b3'
    ctx.fillStyle = card
    ctx.fillRect(0, 0, w, h)
    const clean = xis.filter((v) => typeof v === 'number' && isFinite(v))
    if (!clean.length) {
      ctx.fillStyle = muted
      ctx.fillText('暂无 ξ 数据', 10, 20)
      return
    }
    const minv = Math.min(...clean)
    const maxv = Math.max(...clean)
    const yScale = (v: number) => {
      const denom = maxv - minv || 1
      return h - ((v - minv) / denom) * (h - 20) - 10
    }
    const xScale = (i: number) => (i / (clean.length - 1 || 1)) * (w - 20) + 10
    // grid
    ctx.strokeStyle = border
    ctx.beginPath()
    for (let i = 0; i <= 4; i++) {
      const yy = (i / 4) * (h - 20) + 10
      ctx.moveTo(10, yy)
      ctx.lineTo(w - 10, yy)
    }
    ctx.stroke()
    // line
    ctx.strokeStyle = '#60a5fa' // blue-ish for ξ
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(xScale(0), yScale(clean[0]))
    for (let i = 1; i < clean.length; i++) ctx.lineTo(xScale(i), yScale(clean[i]))
    ctx.stroke()
  }, [xis, themeKey])
  return (
    <div className="chart-box">
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  )
}
