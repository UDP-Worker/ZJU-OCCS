import React from 'react'

export function VoltagePanel() {
  return (
    <fieldset>
      <legend>手动电压</legend>
      <input type="text" placeholder="例如: 0,0,0" defaultValue="0,0,0" />
      <button type="button">应用</button>
    </fieldset>
  )
}

