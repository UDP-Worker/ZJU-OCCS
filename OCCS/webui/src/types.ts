// Shared types aligned with backend models
export type StatusResponse = { running: boolean; iter: number; best_loss: number | null; x?: number[] }
export type WaveformPayload = { lambda: number[]; signal: number[]; target: number[] }
export type ProgressEvent = { iter: number; loss: number; running_min: number; xi?: number | null; kappa?: number | null; gp_max_std?: number | null; x: number[] }

