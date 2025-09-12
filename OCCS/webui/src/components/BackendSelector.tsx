import React from 'react'

export function BackendSelector() {
  return (
    <div>
      <label>
        后端：
        <select defaultValue="mock">
          <option value="mock">mock</option>
          <option value="real" disabled>
            real
          </option>
        </select>
      </label>
    </div>
  )
}

