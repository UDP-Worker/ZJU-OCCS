import React from 'react'

export function StatusBar({ connected, sessionId }: { connected: boolean; sessionId?: string | null }) {
  return (
    <div style={{ marginTop: 12, fontSize: 12, color: connected ? '#7bd88f' : '#e57373' }}>
      状态：{connected ? '已连接' : '未连接'}{sessionId ? `（${sessionId.slice(0, 8)}…）` : ''}
    </div>
  )
}
