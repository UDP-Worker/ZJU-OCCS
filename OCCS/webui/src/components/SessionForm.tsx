import React from 'react'

export function SessionForm() {
  return (
    <fieldset>
      <legend>会话参数</legend>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <label>
          DAC 通道数
          <input type="number" defaultValue={3} min={1} />
        </label>
        <label>
          波长范围 (m)
          <input type="text" defaultValue="1.55e-6..1.56e-6@96" />
        </label>
      </div>
    </fieldset>
  )
}

