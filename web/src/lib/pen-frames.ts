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

export function serializePenFrames(
  penNames: string[],
  frames: PenFrames
): string {
  return penNames
    .filter((name) => frames[name])
    .map((name) => `${name}=${frames[name]}`)
    .join(",")
}
