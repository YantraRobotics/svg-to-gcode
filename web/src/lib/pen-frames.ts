export type PenFrames = Record<string, string>

export const PEN_FRAMES_STORAGE_KEY = "svg-to-gcode.pen-frames"

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
