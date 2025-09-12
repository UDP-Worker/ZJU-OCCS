import React from 'react'

export function DiagnosticsPanel({ xi }: { xi: number | null }) {
  return (
    <fieldset>
      <legend>诊断</legend>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: 12, color: '#98a2b3' }}>xi（探索系数）</div>
          <div style={{ fontSize: 16 }}>{xi != null && isFinite(xi) ? xi.toFixed(4) : '—'}</div>
        </div>
      </div>
    </fieldset>
  )
}

