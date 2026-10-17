export interface PenTool {
  tool: number
  name: string
  hex: string
}

export interface GcodeSegment {
  x0: number
  y0: number
  x1: number
  y1: number
  rapid: boolean
  tool: number | null
}

export interface ParsedGcode {
  tools: Map<number, PenTool>
  segments: GcodeSegment[]
}

export function parsePaperMm(paper: string): [number, number] {
  const upper = paper.trim().toUpperCase()
  if (upper === "A3") return [297, 420]
  if (upper === "A4") return [210, 297]
  if (upper === "A5") return [148, 210]
  if (upper === "LETTER") return [215.9, 279.4]
  const parts = paper.toLowerCase().split("x")
  if (parts.length === 2) {
    const w = Number(parts[0])
    const h = Number(parts[1])
    if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
      return [w, h]
    }
  }
  return [210, 297]
}

export function parseGcode(text: string): ParsedGcode {
  const tools = new Map<number, PenTool>()
  const segments: GcodeSegment[] = []
  let x = 0
  let y = 0
  let z = 5
  let currentTool: number | null = null

  for (const rawLine of text.split(/\r?\n/)) {
    const [codePart, commentPart] = rawLine.split(";", 2)
    if (!codePart.trim() && commentPart) {
      const match = commentPart.trim().match(/^T(\d+)\s+(\S+)\s+(#[0-9A-Fa-f]{6})$/i)
      if (match) {
        const toolNum = Number(match[1])
        tools.set(toolNum, { tool: toolNum, name: match[2], hex: match[3] })
      }
      continue
    }

    const words = codePart.trim().toUpperCase().split(/\s+/).filter(Boolean)
    if (words.length === 0) continue

    for (const w of words) {
      if (w.startsWith("T") && /^\d+$/.test(w.slice(1))) {
        currentTool = Number(w.slice(1))
      }
    }

    const nonTool = words.filter((w) => !w.startsWith("T"))
    if (nonTool.length === 0) continue
    const cmd = nonTool[0]
    if (cmd === "G0" || cmd === "G1") {
      let nx = x
      let ny = y
      let nz = z
      for (const word of nonTool.slice(1)) {
        const axis = word[0]
        const val = Number(word.slice(1))
        if (!Number.isFinite(val)) continue
        if (axis === "X") nx = val
        if (axis === "Y") ny = val
        if (axis === "Z") nz = val
      }
      const rapid = cmd === "G0" || nz > 0 || z > 0
      if (nx !== x || ny !== y) {
        segments.push({
          x0: x,
          y0: y,
          x1: nx,
          y1: ny,
          rapid,
          tool: currentTool,
        })
      }
      x = nx
      y = ny
      z = nz
    }
  }

  return { tools, segments }
}

export const SAMPLE_ROSE_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 240">
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="120" cy="120" r="88" stroke="#231F20" stroke-width="2"/>
    <circle cx="120" cy="120" r="75" stroke="#00843D" stroke-width="1.6"/>
    <circle cx="120" cy="120" r="62" stroke="#00AEEF" stroke-width="1.6"/>
    <path d="M 120 24 L 138 98 L 216 120 L 138 142 L 120 216 L 102 142 L 24 120 L 102 98 Z" fill="#FFF200" stroke="#231F20" stroke-width="1.8"/>
    <path d="M 56 56 L 110 96 L 184 56 L 144 110 L 184 184 L 110 144 L 56 184 L 96 110 Z" stroke="#EC008C" stroke-width="1.6"/>
    <circle cx="120" cy="120" r="16" fill="#EC008C" stroke="#231F20" stroke-width="2"/>
  </g>
</svg>`
