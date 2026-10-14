import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpDown,
  Check,
  Crosshair,
  Download,
  FileCode2,
  Image as ImageIcon,
  Layers,
  Play,
  Plus,
  RefreshCw,
  RotateCw,
  Save,
  Square,
  Upload,
  X,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  SAMPLE_ROSE_SVG,
  parseGcode,
  parsePaperMm,
} from "@/lib/gcode-preview"

interface PenEntry {
  name: string
  hex: string
  nibMm: number
}

type CornerKey = "bl" | "br" | "tr" | "tl"
type CornerCoords = [string, string, string]

const CORNER_LABELS: { key: CornerKey; code: string; title: string }[] = [
  { key: "bl", code: "BL", title: "Bottom-Left" },
  { key: "br", code: "BR", title: "Bottom-Right" },
  { key: "tr", code: "TR", title: "Top-Right" },
  { key: "tl", code: "TL", title: "Top-Left" },
]

interface DrawSummary {
  moves: number
  pen_changes: number
  hops: number
  estimated_seconds: number
  bounds_mm: { min: number[]; max: number[] }
  dry_run: boolean
  worst_error_mm?: number
}

interface ConveyorState {
  running: boolean
  state: string
  direction: "FWD" | "REV"
  port?: string
}

const PEN_PRESETS: Record<string, PenEntry[]> = {
  expo: [
    { name: "red", hex: "#D52B1E", nibMm: 0.8 },
    { name: "green", hex: "#00843D", nibMm: 0.8 },
    { name: "blue", hex: "#1B4F9C", nibMm: 0.8 },
    { name: "black", hex: "#231F20", nibMm: 0.8 },
  ],
  cmyk: [
    { name: "yellow", hex: "#FFF200", nibMm: 0.8 },
    { name: "cyan", hex: "#00AEEF", nibMm: 0.8 },
    { name: "magenta", hex: "#EC008C", nibMm: 0.8 },
    { name: "black", hex: "#231F20", nibMm: 0.8 },
  ],
  mono: [{ name: "black", hex: "#231F20", nibMm: 0.5 }],
}

function serializePens(pens: PenEntry[]): string {
  return pens
    .map(
      (p) =>
        `${p.name.trim()}=${p.hex.toUpperCase()}:${Math.max(0.1, p.nibMm)}`
    )
    .join(",")
}

const CORNERS_STORAGE_KEY = "svg-to-gcode.corners"
const PLACEMENT_MODE_STORAGE_KEY = "svg-to-gcode.placement-mode"

const DEFAULT_CORNERS: Record<CornerKey, CornerCoords> = {
  bl: ["-0.9", "-508.2", "-54.2"],
  br: ["-209.0", "-280.7", "-53.8"],
  tr: ["-345.5", "-423.5", "-55.1"],
  tl: ["-146.8", "-621.3", "-49.1"],
}

function cornerSpanPaper(corners: Record<CornerKey, CornerCoords>): string {
  const pts = CORNER_LABELS.map(({ key }) =>
    corners[key].map((v) => Number(v))
  )
  if (pts.some((p) => p.some((n) => !Number.isFinite(n)))) {
    return "294x200"
  }
  const [bl, br, tr, tl] = pts
  const dist = (a: number[], b: number[]) =>
    Math.hypot(a[0] - b[0], a[1] - b[1])
  const width = Math.floor((dist(bl, br) + dist(tl, tr)) / 2)
  const height = Math.floor((dist(bl, tl) + dist(br, tr)) / 2)
  if (width < 10 || height < 10) {
    return "294x200"
  }
  return `${width}x${height}`
}

function loadSavedCorners(): Record<CornerKey, CornerCoords> {
  try {
    const raw = window.localStorage.getItem(CORNERS_STORAGE_KEY)
    if (!raw) return DEFAULT_CORNERS
    const parsed = JSON.parse(raw) as Partial<Record<CornerKey, unknown>>
    const result = { ...DEFAULT_CORNERS }
    for (const { key } of CORNER_LABELS) {
      const entry = parsed[key]
      if (
        Array.isArray(entry) &&
        entry.length === 3 &&
        entry.every((item) => typeof item === "string")
      ) {
        result[key] = [entry[0], entry[1], entry[2]]
      }
    }
    return result
  } catch {
    return DEFAULT_CORNERS
  }
}

function loadSavedPlacementMode(): "origin" | "corners" {
  try {
    return window.localStorage.getItem(PLACEMENT_MODE_STORAGE_KEY) === "origin"
      ? "origin"
      : "corners"
  } catch {
    return "corners"
  }
}

function serializeCorners(corners: Record<CornerKey, CornerCoords>): string {
  return CORNER_LABELS.flatMap(({ key }) => corners[key]).join(",")
}

function sortCornersAnticlockwise(
  current: Record<CornerKey, CornerCoords>,
  preferLandscape: boolean
): Record<CornerKey, CornerCoords> {
  const entries = CORNER_LABELS.map(({ key }) => ({
    coords: current[key],
    x: Number(current[key][0]),
    y: Number(current[key][1]),
  }))
  if (entries.some((e) => !Number.isFinite(e.x) || !Number.isFinite(e.y))) {
    return current
  }

  const cx = entries.reduce((sum, e) => sum + e.x, 0) / 4
  const cy = entries.reduce((sum, e) => sum + e.y, 0) / 4

  const ccw = [...entries].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  )

  const dist = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    Math.hypot(a.x - b.x, a.y - b.y)

  // Pick the rotation whose bottom edge (BL -> BR) matches the landscape/portrait aspect,
  // breaking ties by keeping BL closest to the current BL row or lowest Y+X.
  const currentBl = entries[0]
  let bestRot = ccw
  let bestScore = Number.POSITIVE_INFINITY

  for (let i = 0; i < 4; i++) {
    const rot = [...ccw.slice(i), ...ccw.slice(0, i)]
    const [bl, br, tr, tl] = rot
    const width = (dist(bl, br) + dist(tl, tr)) / 2
    const height = (dist(bl, tl) + dist(br, tr)) / 2
    const matchesAspect = preferLandscape ? width >= height : height >= width
    const aspectPenalty = matchesAspect ? 0 : 1e6
    const blShift = dist(bl, currentBl)
    const score = aspectPenalty + blShift
    if (score < bestScore) {
      bestScore = score
      bestRot = rot
    }
  }

  return {
    bl: bestRot[0].coords,
    br: bestRot[1].coords,
    tr: bestRot[2].coords,
    tl: bestRot[3].coords,
  }
}

export function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const [serverOnline, setServerOnline] = useState<boolean | null>(null)
  const [sourceName, setSourceName] = useState<string>("compass-rose.svg")
  const [sourceKind, setSourceKind] = useState<"svg" | "raster">("svg")
  const [rawBytes, setRawBytes] = useState<Uint8Array>(() =>
    new TextEncoder().encode(SAMPLE_ROSE_SVG)
  )

  const [penPreset, setPenPreset] = useState<string>("expo")
  const [pens, setPens] = useState<PenEntry[]>(PEN_PRESETS.expo)

  const [traceStyle, setTraceStyle] = useState<string>("centerline")
  const [resolution, setResolution] = useState<number>(1536)
  const [smoothing, setSmoothing] = useState<number>(1)
  const [speckle, setSpeckle] = useState<number>(3)
  const [breaks, setBreaks] = useState<number>(3)

  const [paperPreset, setPaperPreset] = useState<string>("corners")
  const [customPaper, setCustomPaper] = useState<string>("294x200")
  const [margin, setMargin] = useState<number>(12)
  const [hatchSpacing, setHatchSpacing] = useState<number>(0)
  const [hatchAngle, setHatchAngle] = useState<number>(45)
  const [outline, setOutline] = useState<boolean>(true)
  const [feed, setFeed] = useState<number>(1000)
  const [lift, setLift] = useState<number>(6)

  const [placementMode, setPlacementMode] = useState<"origin" | "corners">(
    loadSavedPlacementMode
  )
  const [originX, setOriginX] = useState<string>("5.2")
  const [originY, setOriginY] = useState<string>("-506")
  const [originZ, setOriginZ] = useState<string>("-89.8")
  const [corners, setCorners] =
    useState<Record<CornerKey, CornerCoords>>(loadSavedCorners)
  const [cornersSaved, setCornersSaved] = useState<boolean>(false)
  const [penOffset, setPenOffset] = useState<string>("0")
  const [tilt, setTilt] = useState<number>(0)
  const [rapid, setRapid] = useState<number>(1500)
  const [swapWait, setSwapWait] = useState<number>(5)
  const [robotHost, setRobotHost] = useState<string>("192.168.1.69")

  const [conveyorState, setConveyorState] = useState<ConveyorState | null>(null)
  const [conveyorPort, setConveyorPort] = useState<string>("")
  const [conveyorBusy, setConveyorBusy] = useState<boolean>(false)

  const [tracedSvg, setTracedSvg] = useState<string>(SAMPLE_ROSE_SVG)
  const [gcodeText, setGcodeText] = useState<string>("")
  const [drawSummary, setDrawSummary] = useState<DrawSummary | null>(null)
  const [showRapids, setShowRapids] = useState<boolean>(false)
  const [activeToolFilter, setActiveToolFilter] = useState<number | null>(null)
  const [viewTab, setViewTab] = useState<string>("toolpath")
  const [busy, setBusy] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const cornersPaper = useMemo(() => cornerSpanPaper(corners), [corners])
  const effectivePaper =
    paperPreset === "corners"
      ? cornersPaper
      : paperPreset === "custom"
        ? customPaper
        : paperPreset
  const [paperWidthMm, paperHeightMm] = useMemo(
    () => parsePaperMm(effectivePaper),
    [effectivePaper]
  )

  const parsedGcode = useMemo(() => parseGcode(gcodeText), [gcodeText])

  const buildPlacementParam = useCallback((): Record<string, string> => {
    if (placementMode === "corners") {
      return { corners: serializeCorners(corners) }
    }
    return { origin: `${originX},${originY},${originZ}` }
  }, [placementMode, corners, originX, originY, originZ])

  const runPipeline = useCallback(
    async (
      bytes: Uint8Array,
      kind: "svg" | "raster",
      currentPens: PenEntry[]
    ) => {
      setBusy("Compiling...")
      setErrorMessage(null)
      try {
        const penParam = serializePens(currentPens)
        let currentSvg = ""
        if (kind === "raster") {
          const imgParams = new URLSearchParams({
            pens: penParam,
            style: traceStyle,
            resolution: String(resolution),
            smoothing: String(smoothing),
            speckle: String(speckle),
            breaks: String(breaks),
          })
          const imgRes = await fetch(`/image?${imgParams.toString()}`, {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: bytes as unknown as BodyInit,
          })
          if (!imgRes.ok) {
            const err = await imgRes.json()
            throw new Error(err.error || "Image tracing failed")
          }
          currentSvg = await imgRes.text()
          setTracedSvg(currentSvg)
        } else {
          currentSvg = new TextDecoder().decode(bytes)
          setTracedSvg(currentSvg)
        }

        const gcodeParams = new URLSearchParams({
          paper: effectivePaper,
          pens: penParam,
          margin: String(margin),
          hatch: String(hatchSpacing),
          hatch_angle: String(hatchAngle),
          outline: String(outline),
          feed: String(feed),
          lift: String(lift),
        })
        const gcodeRes = await fetch(`/gcode?${gcodeParams.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "image/svg+xml" },
          body: currentSvg,
        })
        if (!gcodeRes.ok) {
          const err = await gcodeRes.json()
          throw new Error(err.error || "G-code conversion failed")
        }
        const compiledGcode = await gcodeRes.text()
        setGcodeText(compiledGcode)

        const drawParams = new URLSearchParams({
          ...buildPlacementParam(),
          tilt: String(tilt),
          rapid: String(rapid),
          dry_run: "true",
        })
        const drawRes = await fetch(`/draw?${drawParams.toString()}`, {
          method: "POST",
          headers: { "Content-Type": "text/plain" },
          body: compiledGcode,
        })
        if (drawRes.ok) {
          const summary: DrawSummary = await drawRes.json()
          setDrawSummary(summary)
        }
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : String(err))
      } finally {
        setBusy(null)
      }
    },
    [
      traceStyle,
      resolution,
      smoothing,
      speckle,
      breaks,
      effectivePaper,
      margin,
      hatchSpacing,
      hatchAngle,
      outline,
      feed,
      lift,
      buildPlacementParam,
      tilt,
      rapid,
    ]
  )

  useEffect(() => {
    void fetch("/health")
      .then((res) => setServerOnline(res.ok))
      .catch(() => setServerOnline(false))
    void fetch("/conveyor?cmd=STATUS")
      .then(async (res) => {
        if (res.ok) {
          const data = (await res.json()) as ConveyorState
          setConveyorState(data)
        }
      })
      .catch(() => {
        // conveyor serial port not attached yet
      })
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void runPipeline(rawBytes, sourceKind, pens)
    }, 180)
    return () => window.clearTimeout(timer)
  }, [rawBytes, sourceKind, pens, runPipeline])

  const handleFileUpload = async (file: File) => {
    const buffer = await file.arrayBuffer()
    const bytes = new Uint8Array(buffer)
    const isSvg =
      file.type.includes("svg") || file.name.toLowerCase().endsWith(".svg")
    const kind = isSvg ? "svg" : "raster"
    setSourceName(file.name)
    setSourceKind(kind)
    setRawBytes(bytes)
    setActiveToolFilter(null)
  }

  const handleLoadSample = () => {
    const bytes = new TextEncoder().encode(SAMPLE_ROSE_SVG)
    setSourceName("compass-rose.svg")
    setSourceKind("svg")
    setRawBytes(bytes)
    setActiveToolFilter(null)
  }

  const handlePresetChange = (presetKey: string) => {
    setPenPreset(presetKey)
    const preset = PEN_PRESETS[presetKey]
    if (preset) {
      setPens(preset.map((p) => ({ ...p })))
      if (traceStyle === "fill") {
        setHatchSpacing(preset[0].nibMm)
      }
    }
  }

  const handleTraceStyleChange = (nextStyle: string) => {
    setTraceStyle(nextStyle)
    if (nextStyle === "centerline") {
      setResolution(1536)
      setSmoothing(1)
      setSpeckle(3)
      setHatchSpacing(0)
    } else if (nextStyle === "fill") {
      setResolution(400)
      setSmoothing(5)
      setSpeckle(8)
      setHatchSpacing(pens[0]?.nibMm ?? 0.8)
    } else {
      setResolution(140)
      setHatchSpacing(0)
    }
  }

  const handlePlacementModeChange = (mode: "origin" | "corners") => {
    setPlacementMode(mode)
    try {
      window.localStorage.setItem(PLACEMENT_MODE_STORAGE_KEY, mode)
    } catch {
      // ignore storage errors
    }
  }

  const handleSaveCorners = () => {
    try {
      window.localStorage.setItem(CORNERS_STORAGE_KEY, JSON.stringify(corners))
      window.localStorage.setItem(PLACEMENT_MODE_STORAGE_KEY, "corners")
      setCornersSaved(true)
      window.setTimeout(() => setCornersSaved(false), 1800)
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    }
  }

  const handleSortCorners = () => {
    setErrorMessage(null)
    setCorners((prev) =>
      sortCornersAnticlockwise(prev, paperWidthMm >= paperHeightMm)
    )
  }

  const handleRotateCorners = () => {
    setErrorMessage(null)
    setCorners((prev) => ({
      bl: prev.br,
      br: prev.tr,
      tr: prev.tl,
      tl: prev.bl,
    }))
  }

  const handleExecuteOnRobot = async (dryRun: boolean) => {
    if (!gcodeText) return
    setBusy(dryRun ? "Checking reachability..." : "Streaming twists...")
    setErrorMessage(null)
    try {
      const params = new URLSearchParams({
        ...buildPlacementParam(),
        pen_offset: penOffset,
        tilt: String(tilt),
        rapid: String(rapid),
        swap_wait: String(swapWait),
        host: robotHost,
        dry_run: String(dryRun),
      })
      if (activeToolFilter !== null) {
        params.set("tool", String(activeToolFilter))
      }
      const res = await fetch(`/draw?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: gcodeText,
      })
      const payload = await res.json()
      if (!res.ok) {
        throw new Error(payload.error || "Draw request failed")
      }
      setDrawSummary(payload as DrawSummary)
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  const fetchTaughtPoint = async (): Promise<[string, string, string]> => {
    const params = new URLSearchParams({
      pen_offset: penOffset,
      host: robotHost,
    })
    const res = await fetch(`/teach?${params.toString()}`, { method: "POST" })
    const payload = await res.json()
    if (!res.ok) {
      throw new Error(payload.error || "Teach failed")
    }
    const [x, y, z] = payload.origin as [number, number, number]
    return [String(x), String(y), String(z)]
  }

  const handleTeachOrigin = async () => {
    setBusy("Teaching origin...")
    setErrorMessage(null)
    try {
      const [x, y, z] = await fetchTaughtPoint()
      setOriginX(x)
      setOriginY(y)
      setOriginZ(z)
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  const handleConveyorCommand = async (
    cmd: "START" | "STOP" | "FWD" | "REV" | "STATUS"
  ) => {
    setConveyorBusy(true)
    setErrorMessage(null)
    try {
      const params = new URLSearchParams({ cmd })
      if (conveyorPort.trim()) {
        params.set("port", conveyorPort.trim())
      }
      const res = await fetch(`/conveyor?${params.toString()}`, {
        method: "POST",
      })
      const payload = await res.json()
      if (!res.ok) {
        throw new Error(payload.error || "Conveyor command failed")
      }
      setConveyorState(payload as ConveyorState)
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setConveyorBusy(false)
    }
  }

  const handleTeachCorner = async (key: CornerKey) => {
    setBusy(`Teaching ${key.toUpperCase()}...`)
    setErrorMessage(null)
    try {
      const coords = await fetchTaughtPoint()
      setCorners((prev) => ({ ...prev, [key]: coords }))
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  const updateCornerCoord = (
    key: CornerKey,
    axisIndex: 0 | 1 | 2,
    value: string
  ) => {
    setCorners((prev) => {
      const next: CornerCoords = [...prev[key]] as CornerCoords
      next[axisIndex] = value
      return { ...prev, [key]: next }
    })
  }

  const downloadFile = (content: string, filename: string, mime: string) => {
    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const updatePen = (index: number, patch: Partial<PenEntry>) => {
    setPens((prev) =>
      prev.map((item, idx) => (idx === index ? { ...item, ...patch } : item))
    )
  }

  const addPen = () => {
    if (pens.length >= 6) return
    const fallbackNib = pens[pens.length - 1]?.nibMm ?? 0.8
    setPens((prev) => [
      ...prev,
      { name: `pen${prev.length + 1}`, hex: "#1B4F9C", nibMm: fallbackNib },
    ])
  }

  const removePen = (index: number) => {
    if (pens.length <= 1) return
    setPens((prev) => prev.filter((_, idx) => idx !== index))
  }

  const toolPaths = useMemo(() => {
    const nibByName = new Map(pens.map((p) => [p.name.trim(), p.nibMm]))
    const rapidParts: string[] = []
    const byTool = new Map<
      number,
      { color: string; nibMm: number; dParts: string[]; lastX: number; lastY: number }
    >()

    for (const seg of parsedGcode.segments) {
      if (activeToolFilter !== null && seg.tool !== activeToolFilter) continue
      const x0 = seg.x0
      const y0 = paperHeightMm - seg.y0
      const x1 = seg.x1
      const y1 = paperHeightMm - seg.y1

      if (seg.rapid) {
        if (showRapids) {
          rapidParts.push(
            `M${x0.toFixed(2)},${y0.toFixed(2)}L${x1.toFixed(2)},${y1.toFixed(2)}`
          )
        }
        continue
      }

      const toolNum = seg.tool ?? 1
      let group = byTool.get(toolNum)
      if (!group) {
        const info = parsedGcode.tools.get(toolNum)
        const nib = (info ? nibByName.get(info.name) : undefined) ?? 0.8
        group = {
          color: info?.hex || "#231F20",
          nibMm: nib,
          dParts: [],
          lastX: Number.NaN,
          lastY: Number.NaN,
        }
        byTool.set(toolNum, group)
      }
      if (
        !Number.isFinite(group.lastX) ||
        Math.hypot(x0 - group.lastX, y0 - group.lastY) > 1e-3
      ) {
        group.dParts.push(
          `M${x0.toFixed(2)},${y0.toFixed(2)}L${x1.toFixed(2)},${y1.toFixed(2)}`
        )
      } else {
        group.dParts.push(`L${x1.toFixed(2)},${y1.toFixed(2)}`)
      }
      group.lastX = x1
      group.lastY = y1
    }

    return {
      rapidPath: rapidParts.join(""),
      drawGroups: Array.from(byTool.entries()).map(([tool, g]) => ({
        tool,
        color: g.color,
        nibMm: g.nibMm,
        d: g.dParts.join(""),
      })),
    }
  }, [parsedGcode, pens, paperHeightMm, showRapids, activeToolFilter])

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col bg-[#F2EFE9] text-[#231F20]">
      {/* Top Bar */}
      <header className="h-11 border-b border-[#DCD9D0] bg-[#FBFBF9] px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1">
            {pens.map((p, i) => (
              <span
                key={i}
                className="h-2.5 w-2.5 rounded-full border border-black/20"
                style={{ backgroundColor: p.hex }}
              />
            ))}
          </div>
          <Separator orientation="vertical" className="h-4" />
          <h1 className="text-xs font-semibold tracking-tight">
            Owl 6.9 Plotter Workbench
          </h1>
          <Badge variant="secondary" className="font-mono text-[10px]">
            {sourceName}
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-sm border border-[#DCD9D0] bg-[#F2EFE9] px-1.5 py-0.5 text-[10px] font-mono">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                conveyorState?.running
                  ? "bg-[#00843D] animate-pulse"
                  : conveyorState
                    ? "bg-[#6B6661]"
                    : "bg-[#DCD9D0]"
              }`}
            />
            <span className="text-[#6B6661]">Conveyor</span>
            <button
              type="button"
              disabled={conveyorBusy}
              onClick={() =>
                void handleConveyorCommand(
                  conveyorState?.direction === "REV" ? "FWD" : "REV"
                )
              }
              title="Toggle conveyor direction (FWD / REV)"
              className="px-1 py-0.2 rounded-[2px] bg-[#E8E5DC] hover:bg-[#DCD9D0] text-[#231F20] cursor-pointer disabled:opacity-50"
            >
              {conveyorState?.direction ?? "FWD"}
            </button>
            <button
              type="button"
              disabled={conveyorBusy}
              onClick={() =>
                void handleConveyorCommand(
                  conveyorState?.running ? "STOP" : "START"
                )
              }
              className={`inline-flex items-center gap-1 px-1.5 py-0.2 rounded-[2px] font-medium cursor-pointer disabled:opacity-50 ${
                conveyorState?.running
                  ? "bg-[#D52B1E] text-white hover:bg-[#b52217]"
                  : "bg-[#231F20] text-[#FBFBF9] hover:bg-[#3a3537]"
              }`}
            >
              {conveyorState?.running ? (
                <>
                  <Square className="h-2.5 w-2.5 fill-current" />
                  Stop
                </>
              ) : (
                <>
                  <Play className="h-2.5 w-2.5 fill-current" />
                  Start
                </>
              )}
            </button>
          </div>

          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-sm border border-[#DCD9D0] bg-[#F2EFE9] text-[10px] font-mono">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                serverOnline ? "bg-[#00843D]" : "bg-[#D52B1E]"
              }`}
            />
            <span>{serverOnline ? "API ready" : "API offline"}</span>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              downloadFile(
                tracedSvg,
                sourceName.replace(/\.[^.]+$/, "") + ".svg",
                "image/svg+xml"
              )
            }
            disabled={!tracedSvg}
          >
            <Download />
            SVG
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={() =>
              downloadFile(
                gcodeText,
                sourceName.replace(/\.[^.]+$/, "") + ".gcode",
                "text/plain"
              )
            }
            disabled={!gcodeText}
          >
            <Download />
            G-code
          </Button>
        </div>
      </header>

      {/* Main Single-Viewport Grid */}
      <div className="flex-1 grid grid-cols-[296px_1fr_304px] min-h-0 overflow-hidden">
        {/* Left Sidebar: Clean Flat Sections */}
        <aside className="border-r border-[#DCD9D0] bg-[#FBFBF9] px-4 py-3 flex flex-col justify-between min-h-0 overflow-hidden">
          <div className="space-y-3 min-h-0">
            {/* 1. Source */}
            <section className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-semibold">Source File</Label>
                <button
                  type="button"
                  onClick={handleLoadSample}
                  className="text-[11px] text-[#006699] hover:underline cursor-pointer"
                >
                  Load sample
                </button>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.svg"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) void handleFileUpload(file)
                }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  const file = e.dataTransfer.files?.[0]
                  if (file) void handleFileUpload(file)
                }}
                className="w-full border border-dashed border-[#DCD9D0] hover:border-[#231F20] rounded-sm px-3 py-2 flex items-center gap-2.5 cursor-pointer transition-colors bg-[#F2EFE9]/50 text-left"
              >
                <Upload className="h-3.5 w-3.5 text-[#6B6661] shrink-0" />
                <div className="truncate">
                  <div className="text-xs font-medium truncate">
                    {sourceName}
                  </div>
                  <div className="text-[10px] text-[#6B6661]">
                    Click or drop SVG, PNG, JPEG
                  </div>
                </div>
              </button>
            </section>

            <Separator />

            {/* 2. Pens & Per-Pen Nib Size */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-semibold">
                  Pens & Nib (mm)
                </Label>
                <div className="flex items-center gap-2">
                  <Select value={penPreset} onValueChange={handlePresetChange}>
                    <SelectTrigger className="h-6 w-[110px] text-[11px] px-2">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="expo">Expo (RGBK)</SelectItem>
                      <SelectItem value="cmyk">CMYK</SelectItem>
                      <SelectItem value="mono">Monochrome</SelectItem>
                    </SelectContent>
                  </Select>
                  <button
                    type="button"
                    onClick={addPen}
                    disabled={pens.length >= 6}
                    className="inline-flex items-center gap-0.5 text-[11px] text-[#006699] hover:underline cursor-pointer disabled:opacity-40"
                  >
                    <Plus className="h-3 w-3" /> Add
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-1.5">
                {pens.map((pen, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-1.5 rounded-sm border border-[#DCD9D0] bg-[#F2EFE9]/40 px-2 py-1"
                  >
                    <input
                      type="color"
                      value={pen.hex}
                      onChange={(e) => updatePen(idx, { hex: e.target.value })}
                      className="h-3.5 w-3.5 rounded-full border-0 bg-transparent cursor-pointer p-0 shrink-0"
                    />
                    <input
                      value={pen.name}
                      onChange={(e) => updatePen(idx, { name: e.target.value })}
                      className="min-w-0 flex-1 bg-transparent text-[11px] font-mono focus:outline-none truncate"
                    />
                    <input
                      type="number"
                      min={0.1}
                      max={10}
                      step={0.1}
                      value={pen.nibMm}
                      onChange={(e) =>
                        updatePen(idx, {
                          nibMm: Math.max(0.1, Number(e.target.value) || 0.8),
                        })
                      }
                      title="Pen nib width (mm)"
                      className="w-9 h-5 rounded-[2px] border border-[#DCD9D0] bg-[#FBFBF9] px-1 text-right text-[11px] font-mono focus:outline-none focus:border-[#231F20]"
                    />
                    {pens.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removePen(idx)}
                        className="text-[#6B6661] hover:text-[#D52B1E] cursor-pointer shrink-0"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* 3. Raster Image Trace */}
            <Separator />
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-semibold">Image Trace</Label>
                <span className="text-[10px] font-mono text-[#6B6661]">
                  {sourceKind === "raster" ? "Active" : "For PNG/JPG"}
                </span>
              </div>
              <div className="grid grid-cols-[1fr_84px] gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">Mode</Label>
                  <Select
                    value={traceStyle}
                    onValueChange={handleTraceStyleChange}
                  >
                    <SelectTrigger className="h-7 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="centerline">Centerline</SelectItem>
                      <SelectItem value="one-line">One-Line</SelectItem>
                      <SelectItem value="tone">Line Screen</SelectItem>
                      <SelectItem value="fill">Filled Regions</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Resolution (px)
                  </Label>
                  <Input
                    type="number"
                    min={40}
                    max={4000}
                    step={10}
                    value={resolution}
                    onChange={(e) =>
                      setResolution(Number(e.target.value) || 1536)
                    }
                    className="h-7 font-mono text-xs"
                  />
                </div>
              </div>

              {(traceStyle === "centerline" || traceStyle === "fill") && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-[11px] text-[#6B6661]">
                      Smoothing (odd px)
                    </Label>
                    <Input
                      type="number"
                      min={1}
                      max={15}
                      step={2}
                      value={smoothing}
                      onChange={(e) => {
                        const raw = Math.max(1, Number(e.target.value) || 1)
                        setSmoothing(raw % 2 === 0 ? raw + 1 : raw)
                      }}
                      className="h-7 font-mono text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[11px] text-[#6B6661]">
                      Speckle (px)
                    </Label>
                    <Input
                      type="number"
                      min={0}
                      max={50}
                      step={1}
                      value={speckle}
                      onChange={(e) =>
                        setSpeckle(Math.max(0, Number(e.target.value) || 0))
                      }
                      className="h-7 font-mono text-xs"
                    />
                  </div>
                </div>
              )}

              {traceStyle === "one-line" && (
                <div className="space-y-1">
                  <div className="flex justify-between text-[11px] text-[#6B6661]">
                    <span>Max pen lifts</span>
                    <span className="font-mono text-[#231F20]">{breaks}</span>
                  </div>
                  <Slider
                    min={0}
                    max={12}
                    step={1}
                    value={[breaks]}
                    onValueChange={([v]) => setBreaks(v)}
                  />
                </div>
              )}
            </section>

            <Separator />

            {/* 4. Paper & Toolpath */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-semibold">Paper & Toolpath</Label>
                <div className="flex items-center gap-1.5">
                  <Label
                    htmlFor="outline-switch"
                    className="text-[11px] text-[#6B6661] cursor-pointer"
                  >
                    Outline
                  </Label>
                  <Switch
                    id="outline-switch"
                    checked={outline}
                    onCheckedChange={setOutline}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">Paper</Label>
                  <Select value={paperPreset} onValueChange={setPaperPreset}>
                    <SelectTrigger className="h-7 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="corners">
                        4 Corners ({cornersPaper.replace("x", " × ")} mm)
                      </SelectItem>
                      <SelectItem value="294x200">294 × 200 mm</SelectItem>
                      <SelectItem value="250x250">250 × 250 mm</SelectItem>
                      <SelectItem value="A4">A4 (210 × 297)</SelectItem>
                      <SelectItem value="A3">A3 (297 × 420)</SelectItem>
                      <SelectItem value="A5">A5 (148 × 210)</SelectItem>
                      <SelectItem value="LETTER">Letter</SelectItem>
                      <SelectItem value="custom">Custom WxH</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    {paperPreset === "custom" ? "Size (WxH mm)" : "Margin (mm)"}
                  </Label>
                  {paperPreset === "custom" ? (
                    <Input
                      value={customPaper}
                      onChange={(e) => setCustomPaper(e.target.value)}
                      placeholder="294x200"
                      className="h-7 font-mono text-xs"
                    />
                  ) : (
                    <Input
                      type="number"
                      min={0}
                      step={1}
                      value={margin}
                      onChange={(e) => setMargin(Number(e.target.value) || 0)}
                      className="h-7 font-mono text-xs"
                    />
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Default hatch (mm)
                  </Label>
                  <Input
                    type="number"
                    min={0}
                    step={0.1}
                    value={hatchSpacing}
                    onChange={(e) =>
                      setHatchSpacing(Number(e.target.value) || 0)
                    }
                    className="h-7 font-mono text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Hatch angle (deg)
                  </Label>
                  <Input
                    type="number"
                    step={5}
                    value={hatchAngle}
                    onChange={(e) => setHatchAngle(Number(e.target.value) || 0)}
                    className="h-7 font-mono text-xs"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Feed (mm/min)
                  </Label>
                  <Input
                    type="number"
                    min={100}
                    max={6000}
                    step={100}
                    value={feed}
                    onChange={(e) => setFeed(Number(e.target.value) || 1000)}
                    className="h-7 font-mono text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Pen lift Z (mm)
                  </Label>
                  <Input
                    type="number"
                    min={0.5}
                    step={0.5}
                    value={lift}
                    onChange={(e) => setLift(Number(e.target.value) || 6)}
                    className="h-7 font-mono text-xs"
                  />
                </div>
              </div>
            </section>
          </div>

          <Button
            className="w-full shrink-0"
            onClick={() => void runPipeline(rawBytes, sourceKind, pens)}
            disabled={busy !== null}
          >
            <RefreshCw className={busy ? "animate-spin" : ""} />
            {busy ?? "Compile Toolpath"}
          </Button>
        </aside>

        {/* Center Column: Interactive Paper Drafting Bed */}
        <main className="drafting-grid flex flex-col min-h-0 overflow-hidden relative">
          <Tabs
            value={viewTab}
            onValueChange={setViewTab}
            className="flex-1 flex flex-col min-h-0 overflow-hidden"
          >
            <div className="h-9 border-b border-[#DCD9D0] bg-[#FBFBF9]/90 backdrop-blur-xs px-3 flex items-center justify-between shrink-0">
              <TabsList className="h-7">
                <TabsTrigger value="toolpath" className="gap-1.5 text-[11px] py-0.5">
                  <Layers className="h-3 w-3" />
                  Toolpath
                </TabsTrigger>
                <TabsTrigger value="svg" className="gap-1.5 text-[11px] py-0.5">
                  <ImageIcon className="h-3 w-3" />
                  SVG
                </TabsTrigger>
                <TabsTrigger value="raw" className="gap-1.5 text-[11px] py-0.5">
                  <FileCode2 className="h-3 w-3" />
                  G-code ({gcodeText ? gcodeText.split("\n").length : 0})
                </TabsTrigger>
              </TabsList>

              {viewTab === "toolpath" && (
                <div className="flex items-center gap-1.5">
                  <Switch
                    id="rapids-toggle"
                    checked={showRapids}
                    onCheckedChange={setShowRapids}
                  />
                  <Label
                    htmlFor="rapids-toggle"
                    className="text-[11px] text-[#6B6661] cursor-pointer"
                  >
                    Show rapid travel
                  </Label>
                </div>
              )}
            </div>

            {errorMessage && (
              <div className="mx-3 mt-2 rounded-sm border border-[#D52B1E] bg-[#D52B1E]/10 px-2.5 py-1.5 text-[11px] text-[#D52B1E] font-mono shrink-0">
                {errorMessage}
              </div>
            )}

            {/* Toolpath Canvas */}
            <TabsContent
              value="toolpath"
              className="flex-1 flex flex-col items-center justify-between p-4 mt-0 min-h-0 overflow-hidden"
            >
              <div className="flex-1 flex items-center justify-center w-full min-h-0 overflow-hidden">
                <svg
                  viewBox={`0 0 ${paperWidthMm} ${paperHeightMm}`}
                  className="max-w-full max-h-full w-auto h-full bg-[#FBFBF9] shadow-sm border border-[#231F20]/40"
                  style={{
                    aspectRatio: `${paperWidthMm} / ${paperHeightMm}`,
                  }}
                >
                  {margin > 0 &&
                    paperWidthMm > margin * 2 &&
                    paperHeightMm > margin * 2 && (
                      <rect
                        x={margin}
                        y={margin}
                        width={paperWidthMm - margin * 2}
                        height={paperHeightMm - margin * 2}
                        fill="none"
                        stroke="#DCD9D0"
                        strokeWidth={Math.max(paperWidthMm, paperHeightMm) * 0.0025}
                        strokeDasharray="3 3"
                      />
                    )}

                  <g
                    stroke="#D52B1E"
                    strokeWidth={Math.max(paperWidthMm, paperHeightMm) * 0.003}
                  >
                    <line
                      x1={0}
                      y1={paperHeightMm}
                      x2={Math.min(15, paperWidthMm * 0.08)}
                      y2={paperHeightMm}
                    />
                    <line
                      x1={0}
                      y1={paperHeightMm}
                      x2={0}
                      y2={paperHeightMm - Math.min(15, paperHeightMm * 0.08)}
                    />
                  </g>

                  {toolPaths.rapidPath && (
                    <path
                      d={toolPaths.rapidPath}
                      fill="none"
                      stroke="#9E9890"
                      strokeWidth={Math.max(paperWidthMm, paperHeightMm) * 0.0015}
                      strokeDasharray="2 2"
                      opacity={0.4}
                    />
                  )}

                  {toolPaths.drawGroups.map((group) => (
                    <path
                      key={group.tool}
                      d={group.d}
                      fill="none"
                      stroke={group.color}
                      strokeWidth={group.nibMm}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      opacity={0.95}
                    />
                  ))}
                </svg>
              </div>

              {/* Pen Tool Filter Bar */}
              <div className="mt-2.5 flex flex-wrap items-center gap-1.5 bg-[#FBFBF9] border border-[#DCD9D0] px-2.5 py-1 rounded-sm shrink-0">
                <button
                  type="button"
                  onClick={() => setActiveToolFilter(null)}
                  className={`px-2 py-0.5 text-[10px] font-mono rounded-[2px] cursor-pointer ${
                    activeToolFilter === null
                      ? "bg-[#231F20] text-[#FBFBF9]"
                      : "bg-[#E8E5DC] text-[#231F20]"
                  }`}
                >
                  All Tools
                </button>
                {Array.from(parsedGcode.tools.values()).map((tool) => (
                  <button
                    key={tool.tool}
                    type="button"
                    onClick={() =>
                      setActiveToolFilter(
                        activeToolFilter === tool.tool ? null : tool.tool
                      )
                    }
                    className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] font-mono rounded-[2px] border cursor-pointer ${
                      activeToolFilter === tool.tool
                        ? "border-[#231F20] bg-[#231F20] text-[#FBFBF9]"
                        : "border-[#DCD9D0] bg-[#FBFBF9] text-[#231F20]"
                    }`}
                  >
                    <span
                      className="h-2 w-2 rounded-full border border-black/20"
                      style={{ backgroundColor: tool.hex }}
                    />
                    T{tool.tool} {tool.name}
                  </button>
                ))}
              </div>
            </TabsContent>

            {/* Traced SVG View */}
            <TabsContent
              value="svg"
              className="flex-1 flex items-center justify-center p-4 mt-0 min-h-0 overflow-hidden"
            >
              <div
                className="max-w-full max-h-full h-full flex items-center justify-center bg-[#FBFBF9] p-3 border border-[#DCD9D0] shadow-xs overflow-hidden [&>svg]:max-h-full [&>svg]:max-w-full [&>svg]:w-auto [&>svg]:h-full"
                dangerouslySetInnerHTML={{ __html: tracedSvg }}
              />
            </TabsContent>

            {/* Raw G-code View */}
            <TabsContent
              value="raw"
              className="flex-1 p-4 mt-0 min-h-0 overflow-hidden"
            >
              <pre className="w-full h-full overflow-auto rounded-sm border border-[#DCD9D0] bg-[#FBFBF9] p-3 text-[11px] font-mono leading-relaxed text-[#231F20]">
                {gcodeText || "; No G-code compiled yet"}
              </pre>
            </TabsContent>
          </Tabs>
        </main>

        {/* Right Sidebar: Telemetry & Arm Placement */}
        <aside className="border-l border-[#DCD9D0] bg-[#FBFBF9] px-4 py-3 flex flex-col justify-between min-h-0 overflow-hidden">
          <div className="space-y-3 min-h-0">
            {/* Motion Summary */}
            <section className="space-y-2">
              <Label className="text-xs font-semibold block">
                Motion Summary
              </Label>
              {drawSummary ? (
                <>
                  <div className="grid grid-cols-3 gap-1.5">
                    <div className="rounded-sm bg-[#F2EFE9]/70 px-2 py-1 border border-[#DCD9D0]">
                      <div className="text-[10px] text-[#6B6661]">Moves</div>
                      <div className="text-xs font-mono font-medium">
                        {drawSummary.moves.toLocaleString()}
                      </div>
                    </div>
                    <div className="rounded-sm bg-[#F2EFE9]/70 px-2 py-1 border border-[#DCD9D0]">
                      <div className="text-[10px] text-[#6B6661]">Swaps</div>
                      <div className="text-xs font-mono font-medium">
                        {drawSummary.pen_changes}
                      </div>
                    </div>
                    <div className="rounded-sm bg-[#F2EFE9]/70 px-2 py-1 border border-[#DCD9D0]">
                      <div className="text-[10px] text-[#6B6661]">Time</div>
                      <div className="text-xs font-mono font-medium">
                        {(drawSummary.estimated_seconds / 60).toFixed(1)}m
                      </div>
                    </div>
                  </div>

                  <div className="rounded-sm bg-[#F2EFE9]/70 px-2.5 py-1.5 border border-[#DCD9D0] space-y-0.5 font-mono text-[11px]">
                    <div className="text-[10px] font-sans text-[#6B6661]">
                      Base-Frame Envelope
                    </div>
                    {["X", "Y", "Z"].map((axis, i) => (
                      <div key={axis} className="flex justify-between">
                        <span className="text-[#6B6661]">{axis}</span>
                        <span>
                          {drawSummary.bounds_mm.min[i].toFixed(1)} ..{" "}
                          {drawSummary.bounds_mm.max[i].toFixed(1)} mm
                        </span>
                      </div>
                    ))}
                  </div>

                  {drawSummary.worst_error_mm !== undefined && (
                    <div className="rounded-sm border border-[#00843D] bg-[#00843D]/10 px-2.5 py-1 text-[11px] font-mono text-[#00843D]">
                      Done · Max error: {drawSummary.worst_error_mm.toFixed(2)} mm
                    </div>
                  )}
                </>
              ) : (
                <p className="text-xs text-[#6B6661]">
                  Compile a drawing to inspect bounds and duration.
                </p>
              )}
            </section>

            <Separator />

            {/* Owl 6.9 Arm Placement */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs font-semibold">Arm Placement</Label>
                <div className="inline-flex rounded-sm border border-[#DCD9D0] bg-[#F2EFE9] p-0.5 text-[10px] font-medium">
                  <button
                    type="button"
                    onClick={() => handlePlacementModeChange("origin")}
                    className={`px-2 py-0.5 rounded-[2px] cursor-pointer transition-colors ${
                      placementMode === "origin"
                        ? "bg-[#231F20] text-[#FBFBF9]"
                        : "text-[#6B6661] hover:text-[#231F20]"
                    }`}
                  >
                    Origin
                  </button>
                  <button
                    type="button"
                    onClick={() => handlePlacementModeChange("corners")}
                    className={`px-2 py-0.5 rounded-[2px] cursor-pointer transition-colors ${
                      placementMode === "corners"
                        ? "bg-[#231F20] text-[#FBFBF9]"
                        : "text-[#6B6661] hover:text-[#231F20]"
                    }`}
                  >
                    4 Corners
                  </button>
                </div>
              </div>

              {placementMode === "origin" ? (
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-[11px] text-[#6B6661]">
                      Origin X, Y, Z (base mm)
                    </Label>
                    <button
                      type="button"
                      onClick={() => void handleTeachOrigin()}
                      disabled={busy !== null}
                      className="inline-flex items-center gap-1 text-[11px] text-[#006699] hover:underline cursor-pointer disabled:opacity-50"
                    >
                      <Crosshair className="h-3 w-3" /> Teach
                    </button>
                  </div>
                  <div className="grid grid-cols-3 gap-1.5">
                    <Input
                      value={originX}
                      onChange={(e) => setOriginX(e.target.value)}
                      placeholder="X"
                      className="h-7 font-mono text-xs"
                    />
                    <Input
                      value={originY}
                      onChange={(e) => setOriginY(e.target.value)}
                      placeholder="Y"
                      className="h-7 font-mono text-xs"
                    />
                    <Input
                      value={originZ}
                      onChange={(e) => setOriginZ(e.target.value)}
                      placeholder="Z"
                      className="h-7 font-mono text-xs"
                    />
                  </div>
                </div>
              ) : (
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-[#6B6661]">
                      Corners X, Y, Z (mm)
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleSortCorners}
                        title="Sort the 4 points anticlockwise into BL, BR, TR, TL"
                        className="inline-flex items-center gap-0.5 text-[11px] text-[#006699] hover:underline cursor-pointer"
                      >
                        <ArrowUpDown className="h-3 w-3" />
                        <span>Sort</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleRotateCorners}
                        title="Cycle corners 90° anticlockwise (BL ← BR ← TR ← TL)"
                        className="inline-flex items-center gap-0.5 text-[11px] text-[#006699] hover:underline cursor-pointer"
                      >
                        <RotateCw className="h-3 w-3" />
                        <span>90°</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleSaveCorners}
                        className="inline-flex items-center gap-0.5 text-[11px] text-[#006699] hover:underline cursor-pointer"
                      >
                        {cornersSaved ? (
                          <>
                            <Check className="h-3 w-3 text-[#00843D]" />
                            <span className="text-[#00843D]">Saved</span>
                          </>
                        ) : (
                          <>
                            <Save className="h-3 w-3" />
                            <span>Save</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                  {CORNER_LABELS.map(({ key, code, title }) => (
                    <div
                      key={key}
                      className="grid grid-cols-[24px_1fr_1fr_1fr_24px] gap-1 items-center"
                    >
                      <span
                        title={title}
                        className="text-[11px] font-mono font-semibold text-[#6B6661]"
                      >
                        {code}
                      </span>
                      {([0, 1, 2] as const).map((axisIdx) => (
                        <Input
                          key={axisIdx}
                          value={corners[key][axisIdx]}
                          onChange={(e) =>
                            updateCornerCoord(key, axisIdx, e.target.value)
                          }
                          className="h-6 px-1.5 font-mono text-[11px]"
                        />
                      ))}
                      <button
                        type="button"
                        onClick={() => void handleTeachCorner(key)}
                        disabled={busy !== null}
                        title={`Teach ${title} (${code}) from arm`}
                        className="h-6 w-6 inline-flex items-center justify-center rounded-sm border border-[#DCD9D0] bg-[#F2EFE9]/60 text-[#006699] hover:border-[#006699] cursor-pointer disabled:opacity-40"
                      >
                        <Crosshair className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Pen offset (mm)
                  </Label>
                  <Input
                    value={penOffset}
                    onChange={(e) => setPenOffset(e.target.value)}
                    className="h-7 font-mono text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Tilt (deg)
                  </Label>
                  <Input
                    type="number"
                    min={0}
                    max={30}
                    value={tilt}
                    onChange={(e) => setTilt(Number(e.target.value) || 0)}
                    className="h-7 font-mono text-xs"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Rapid (mm/min)
                  </Label>
                  <Input
                    type="number"
                    min={500}
                    max={6000}
                    step={250}
                    value={rapid}
                    onChange={(e) => setRapid(Number(e.target.value) || 1500)}
                    className="h-7 font-mono text-xs"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-[11px] text-[#6B6661]">
                    Swap wait (s)
                  </Label>
                  <Input
                    type="number"
                    min={0}
                    step={1}
                    value={swapWait}
                    onChange={(e) => setSwapWait(Number(e.target.value) || 0)}
                    className="h-7 font-mono text-xs"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <Label className="text-[11px] text-[#6B6661]">
                  Controller ZMQ host
                </Label>
                <Input
                  value={robotHost}
                  onChange={(e) => setRobotHost(e.target.value)}
                  className="h-7 font-mono text-xs"
                />
              </div>

              <Separator />

              {/* ESP32 Conveyor Belt Control */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs font-semibold">Conveyor Belt</Label>
                  <span className="text-[10px] font-mono text-[#6B6661]">
                    {conveyorState
                      ? `${conveyorState.state} · ${conveyorState.direction}`
                      : "ESP32 UART"}
                  </span>
                </div>

                <div className="grid grid-cols-[1fr_auto_auto] gap-1.5 items-center">
                  <Input
                    value={conveyorPort}
                    onChange={(e) => setConveyorPort(e.target.value)}
                    placeholder={conveyorState?.port || "Auto USB port"}
                    title="ESP32 serial port (leave blank to auto-detect)"
                    className="h-7 font-mono text-[11px]"
                  />
                  <div className="inline-flex rounded-sm border border-[#DCD9D0] bg-[#F2EFE9] p-0.5 text-[10px] font-mono">
                    <button
                      type="button"
                      disabled={conveyorBusy}
                      onClick={() => void handleConveyorCommand("FWD")}
                      title="Forward direction"
                      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-[2px] cursor-pointer ${
                        (conveyorState?.direction ?? "FWD") === "FWD"
                          ? "bg-[#231F20] text-[#FBFBF9]"
                          : "text-[#6B6661] hover:text-[#231F20]"
                      }`}
                    >
                      <ArrowRight className="h-2.5 w-2.5" />
                      FWD
                    </button>
                    <button
                      type="button"
                      disabled={conveyorBusy}
                      onClick={() => void handleConveyorCommand("REV")}
                      title="Reverse direction"
                      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-[2px] cursor-pointer ${
                        conveyorState?.direction === "REV"
                          ? "bg-[#231F20] text-[#FBFBF9]"
                          : "text-[#6B6661] hover:text-[#231F20]"
                      }`}
                    >
                      <ArrowLeft className="h-2.5 w-2.5" />
                      REV
                    </button>
                  </div>
                  <Button
                    size="sm"
                    variant={conveyorState?.running ? "destructive" : "default"}
                    disabled={conveyorBusy}
                    onClick={() =>
                      void handleConveyorCommand(
                        conveyorState?.running ? "STOP" : "START"
                      )
                    }
                    className={`h-7 px-2.5 text-xs ${
                      conveyorState?.running
                        ? "bg-[#D52B1E] hover:bg-[#b52217] text-white"
                        : ""
                    }`}
                  >
                    {conveyorState?.running ? (
                      <>
                        <Square className="h-3 w-3 fill-current" />
                        Stop
                      </>
                    ) : (
                      <>
                        <Play className="h-3 w-3 fill-current" />
                        Start
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </section>
          </div>

          <div className="space-y-1.5 shrink-0">
            <Button
              variant="outline"
              className="w-full"
              onClick={() => void handleExecuteOnRobot(true)}
              disabled={!gcodeText || busy !== null}
            >
              Verify Dry-Run Envelope
            </Button>
            <Button
              variant="default"
              className="w-full bg-[#006699] hover:bg-[#005580] text-white"
              onClick={() => void handleExecuteOnRobot(false)}
              disabled={!gcodeText || busy !== null}
            >
              <Play />
              {activeToolFilter !== null
                ? `Draw Only T${activeToolFilter}`
                : "Draw on Owl 6.9 Arm"}
            </Button>
          </div>
        </aside>
      </div>
    </div>
  )
}

export default App
