import React from 'react'

export function OptimizerControls() {
  return (
    <fieldset>
      <legend>优化控制</legend>
      <label>
        迭代次数
        <input type="number" defaultValue={10} min={1} />
      </label>
      <button type="button">开始优化</button>
      <button type="button">停止</button>
    </fieldset>
  )
}

