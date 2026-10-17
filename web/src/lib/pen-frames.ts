export type PenFrames = Record<string, string>

export const PEN_FRAMES_STORAGE_KEY = "svg-to-gcode.pen-frames"

export interface TargetFrames {
  current: string
  eligible: string[]
}

export async function listTargetFrames(host: string): Promise<TargetFrames> {
  const params = new URLSearchParams({ host })
  const res = await fetch(`/frames?${params.toString()}`)
  const payload = await res.json()
  if (!res.ok) {
    throw new Error(payload.error || "Could not list target frames")
  }
  return payload as TargetFrames
}

export function loadSavedPenFrames(): PenFrames {
  try {
    const raw = window.localStorage.getItem(PEN_FRAMES_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, unknown>
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string] =>
          typeof entry[1] === "string" && entry[1] !== ""
      )
    )
  } catch {
    return {}
  }
}

export type FramePoint = [string, string, string]
export type FramePoints = Record<string, FramePoint>

export const FRAME_POINTS_STORAGE_KEY = "svg-to-gcode.frame-points"
export const REFERENCE_FRAME = "target_frame_1"
export const DEFAULT_TOUCH_FRAMES = [
  REFERENCE_FRAME,
  "target_frame_2",
  "target_frame_3",
]

export function loadSavedFramePoints(): FramePoints {
  try {
    const raw = window.localStorage.getItem(FRAME_POINTS_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, unknown>
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, FramePoint] =>
          Array.isArray(entry[1]) &&
          entry[1].length === 3 &&
          entry[1].every((value) => typeof value === "string")
      )
    )
  } catch {
    return {}
  }
}

function isFilled(point: FramePoint | undefined): point is FramePoint {
  return point !== undefined && point.some((value) => value.trim() !== "")
}

export function serializeFramePoints(points: FramePoints): string {
  return Object.entries(points)
    .filter(([, point]) => isFilled(point))
    .map(([frame, point]) => `${frame}=${point.map((v) => v.trim()).join(",")}`)
    .join(";")
}

export function frameShift(
  points: FramePoints,
  frame: string
): [number, number, number] | null {
  const reference = points[REFERENCE_FRAME]
  const point = points[frame]
  if (!isFilled(reference) || !isFilled(point)) return null
  const shift = point.map((value, axis) => Number(value) - Number(reference[axis]))
  return shift.every(Number.isFinite)
    ? (shift as [number, number, number])
    : null
}

export function serializePenFrames(
  penNames: string[],
  frames: PenFrames
): string {
  return penNames
    .filter((name) => frames[name])
    .map((name) => `${name}=${frames[name]}`)
    .join(",")
}
