import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Download,
  FileCode2,
  Image as ImageIcon,
  Layers,
  Play,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
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
}

interface DrawSummary {
  moves: number
  pen_changes: number
  hops: number
  estimated_seconds: number
  bounds_mm: { min: number[]; max: number[] }
  dry_run: boolean
  worst_error_mm?: number
}

const PEN_PRESETS: Record<string, PenEntry[]> = {
  cmyk: [
    { name: "yellow", hex: "#FFF200" },
    { name: "cyan", hex: "#00AEEF" },
    { name: "magenta", hex: "#EC008C" },
    { name: "black", hex: "#231F20" },
  ],
  expo: [
    { name: "red", hex: "#D52B1E" },
    { name: "green", hex: "#00843D" },
    { name: "blue", hex: "#1B4F9C" },
    { name: "black", hex: "#231F20" },
  ],
  mono: [{ name: "black", hex: "#231F20" }],
}

function serializePens(pens: PenEntry[]): string {
  return pens.map((p) => `${p.name.trim()}=${p.hex.toUpperCase()}`).join(",")
}

export function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const [serverOnline, setServerOnline] = useState<boolean | null>(null)
  const [sourceName, setSourceName] = useState<string>("compass-rose.svg")
  const [sourceKind, setSourceKind] = useState<"svg" | "raster">("svg")
  const [rawBytes, setRawBytes] = useState<Uint8Array>(() =>
    new TextEncoder().encode(SAMPLE_ROSE_SVG)
  )

  const [pens, setPens] = useState<PenEntry[]>(PEN_PRESETS.cmyk)
  const [traceStyle, setTraceStyle] = useState<string>("one-line")
  const [resolution, setResolution] = useState<number>(140)
  const [smoothing, setSmoothing] = useState<number>(5)
  const [speckle, setSpeckle] = useState<number>(8)
  const [blend, setBlend] = useState<number>(2)
  const [breaks, setBreaks] = useState<number>(3)

  const [paperPreset, setPaperPreset] = useState<string>("250x250")
  const [customPaper, setCustomPaper] = useState<string>("250x250")
  const [margin, setMargin] = useState<number>(12)
  const [hatchSpacing, setHatchSpacing] = useState<number>(0.8)
  const [hatchAngle, setHatchAngle] = useState<number>(45)
  const [outline, setOutline] = useState<boolean>(true)
  const [feed, setFeed] = useState<number>(1000)
  const [lift, setLift] = useState<number>(2)

  const [originX, setOriginX] = useState<string>("-125")
  const [originY, setOriginY] = useState<string>("-460")
  const [originZ, setOriginZ] = useState<string>("30.6")
  const [penOffset, setPenOffset] = useState<string>("40")
  const [tilt, setTilt] = useState<number>(0)
  const [rapid, setRapid] = useState<number>(3000)
  const [swapWait, setSwapWait] = useState<number>(5)
  const [robotHost, setRobotHost] = useState<string>("127.0.0.1")

  const [tracedSvg, setTracedSvg] = useState<string>(SAMPLE_ROSE_SVG)
  const [gcodeText, setGcodeText] = useState<string>("")
  const [drawSummary, setDrawSummary] = useState<DrawSummary | null>(null)
  const [showRapids, setShowRapids] = useState<boolean>(true)
  const [activeToolFilter, setActiveToolFilter] = useState<number | null>(null)
  const [viewTab, setViewTab] = useState<string>("toolpath")
  const [busy, setBusy] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const effectivePaper = paperPreset === "custom" ? customPaper : paperPreset
  const [paperWidthMm, paperHeightMm] = useMemo(
    () => parsePaperMm(effectivePaper),
    [effectivePaper]
  )

  const parsedGcode = useMemo(() => parseGcode(gcodeText), [gcodeText])

  const runPipeline = useCallback(
    async (
      bytes: Uint8Array,
      kind: "svg" | "raster",
      currentPens: PenEntry[]
    ) => {
      setBusy("Compiling toolpath...")
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
            blend: String(blend),
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
          origin: `${originX},${originY},${originZ}`,
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
      blend,
      breaks,
      effectivePaper,
      margin,
      hatchSpacing,
      hatchAngle,
      outline,
      feed,
      lift,
      originX,
      originY,
      originZ,
      tilt,
      rapid,
    ]
  )

  const initializedRef = useRef(false)
  useEffect(() => {
    if (initializedRef.current) return
    initializedRef.current = true
    void fetch("/health")
      .then((res) => setServerOnline(res.ok))
      .catch(() => setServerOnline(false))
    void runPipeline(rawBytes, sourceKind, pens)
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
    await runPipeline(bytes, kind, pens)
  }

  const handleLoadSample = async () => {
    const bytes = new TextEncoder().encode(SAMPLE_ROSE_SVG)
    setSourceName("compass-rose.svg")
    setSourceKind("svg")
    setRawBytes(bytes)
    setActiveToolFilter(null)
    await runPipeline(bytes, "svg", pens)
  }

  const handleExecuteOnRobot = async (dryRun: boolean) => {
    if (!gcodeText) return
    setBusy(dryRun ? "Checking reachability..." : "Streaming twists to Owl 6.9...")
    setErrorMessage(null)
    try {
      const params = new URLSearchParams({
        origin: `${originX},${originY},${originZ}`,
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

  const handleTeachOrigin = async () => {
    setBusy("Reading pen tip from controller...")
    setErrorMessage(null)
    try {
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
      setOriginX(String(x))
      setOriginY(String(y))
      setOriginZ(String(z))
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
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
    setPens((prev) => [...prev, { name: `pen${prev.length + 1}`, hex: "#1B4F9C" }])
  }

  const removePen = (index: number) => {
    if (pens.length <= 1) return
    setPens((prev) => prev.filter((_, idx) => idx !== index))
  }

  const visibleSegments = useMemo(() => {
    return parsedGcode.segments.filter((seg) => {
      if (!showRapids && seg.rapid) return false
      if (activeToolFilter !== null && seg.tool !== activeToolFilter) return false
      return true
    })
  }, [parsedGcode.segments, showRapids, activeToolFilter])

  return (
    <div className="min-h-screen flex flex-col bg-[#F2EFE9] text-[#231F20]">
      {/* Top Bar */}
      <header className="h-12 border-b border-[#DCD9D0] bg-[#FBFBF9] px-4 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-[#FFF200] border border-[#231F20]/30" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#00AEEF]" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#EC008C]" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#231F20]" />
          </div>
          <Separator orientation="vertical" className="h-4" />
          <h1 className="text-sm font-semibold tracking-tight">
            Owl 6.9 Plotter Workbench
          </h1>
          <Badge variant="secondary" className="font-mono text-[11px]">
            {sourceName}
          </Badge>
          <Badge variant="outline" className="font-mono text-[11px]">
            {paperWidthMm} × {paperHeightMm} mm
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-sm border border-[#DCD9D0] bg-[#F2EFE9] text-[11px] font-mono">
            <span
              className={`h-2 w-2 rounded-full ${
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
            Export SVG
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
            Download G-code
          </Button>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-[360px_1fr_320px] min-h-0">
        {/* Left Column: Source, Pens & Conversion Controls */}
        <aside className="border-r border-[#DCD9D0] bg-[#FBFBF9] p-4 space-y-4 overflow-y-auto">
          {/* File Dropzone */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle>Artwork Source</CardTitle>
                <button
                  type="button"
                  onClick={handleLoadSample}
                  className="text-[11px] text-[#006699] hover:underline cursor-pointer"
                >
                  Load sample SVG
                </button>
              </div>
            </CardHeader>
            <CardContent className="space-y-2.5">
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
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  const file = e.dataTransfer.files?.[0]
                  if (file) void handleFileUpload(file)
                }}
                className="border border-dashed border-[#DCD9D0] hover:border-[#231F20] rounded-sm p-3 text-center cursor-pointer transition-colors bg-[#F2EFE9]/50"
              >
                <Upload className="h-4 w-4 mx-auto mb-1 text-[#6B6661]" />
                <p className="text-xs font-medium">
                  Drop an SVG or photo, or click to browse
                </p>
                <p className="text-[11px] text-[#6B6661] mt-0.5">
                  PNG, JPEG, WebP, or vector SVG
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Pen Carousel */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle>Pen Set (lightest drawn first)</CardTitle>
                <div className="flex gap-1">
                  {(["cmyk", "expo", "mono"] as const).map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => setPens(PEN_PRESETS[preset])}
                      className="px-1.5 py-0.5 text-[10px] font-mono rounded-[2px] border border-[#DCD9D0] bg-[#F2EFE9] hover:bg-[#E8E5DC] uppercase cursor-pointer"
                    >
                      {preset}
                    </button>
                  ))}
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-2">
              {pens.map((pen, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <input
                    type="color"
                    value={pen.hex}
                    onChange={(e) => updatePen(idx, { hex: e.target.value })}
                    className="h-7 w-8 rounded-sm border border-[#DCD9D0] bg-transparent cursor-pointer p-0.5"
                  />
                  <Input
                    value={pen.name}
                    onChange={(e) => updatePen(idx, { name: e.target.value })}
                    className="h-7 text-xs font-mono"
                    placeholder="pen-name"
                  />
                  <span className="text-[11px] font-mono text-[#6B6661] w-16">
                    {pen.hex.toUpperCase()}
                  </span>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => removePen(idx)}
                    disabled={pens.length <= 1}
                  >
                    <Trash2 className="h-3.5 w-3.5 text-[#6B6661]" />
                  </Button>
                </div>
              ))}
              <Button
                variant="outline"
                size="sm"
                className="w-full mt-1"
                onClick={addPen}
              >
                <Plus />
                Add Pen
              </Button>
            </CardContent>
          </Card>

          {/* Raster Tracing Options (shown when raster or always available) */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle>Image Trace Style</CardTitle>
                <Badge variant="secondary" className="text-[10px] font-mono">
                  {sourceKind === "raster" ? "Active" : "Raster only"}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1.5">
                <Label>Tracing Mode</Label>
                <Select value={traceStyle} onValueChange={setTraceStyle}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fill">Filled Regions (Logos & Flat Art)</SelectItem>
                    <SelectItem value="tone">Line Screen Shading (Photos)</SelectItem>
                    <SelectItem value="one-line">Continuous One-Line (Photos)</SelectItem>
                    <SelectItem value="centerline">Centerline Skeleton (Line Art)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <Label>Resolution (long side)</Label>
                  <span className="font-mono text-[11px] text-[#6B6661]">
                    {resolution} px
                  </span>
                </div>
                <Slider
                  min={40}
                  max={600}
                  step={10}
                  value={[resolution]}
                  onValueChange={([v]) => setResolution(v)}
                />
              </div>

              {traceStyle === "one-line" && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <Label>Max Pen Lifts (Breaks)</Label>
                    <span className="font-mono text-[11px] text-[#6B6661]">
                      {breaks}
                    </span>
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

              {(traceStyle === "fill" || traceStyle === "centerline") && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label>Smoothing (odd px)</Label>
                    <Input
                      type="number"
                      min={1}
                      step={2}
                      value={smoothing}
                      onChange={(e) => setSmoothing(Number(e.target.value) || 1)}
                      className="font-mono"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Speckle (px)</Label>
                    <Input
                      type="number"
                      min={0}
                      step={1}
                      value={speckle}
                      onChange={(e) => setSpeckle(Number(e.target.value) || 0)}
                      className="font-mono"
                    />
                  </div>
                </div>
              )}

              {traceStyle === "fill" && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <Label>Max Overlapped Pens (Blend)</Label>
                    <span className="font-mono text-[11px] text-[#6B6661]">
                      {blend}
                    </span>
                  </div>
                  <Slider
                    min={1}
                    max={4}
                    step={1}
                    value={[blend]}
                    onValueChange={([v]) => setBlend(v)}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Paper & G-code Parameters */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Paper & Toolpath Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label>Paper Size</Label>
                  <Select value={paperPreset} onValueChange={setPaperPreset}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="250x250">250 × 250 mm Square</SelectItem>
                      <SelectItem value="A4">A4 (210 × 297 mm)</SelectItem>
                      <SelectItem value="A3">A3 (297 × 420 mm)</SelectItem>
                      <SelectItem value="A5">A5 (148 × 210 mm)</SelectItem>
                      <SelectItem value="LETTER">Letter (215.9 × 279.4)</SelectItem>
                      <SelectItem value="custom">Custom WxH...</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label>Margin (mm)</Label>
                  <Input
                    type="number"
                    min={0}
                    step={1}
                    value={margin}
                    onChange={(e) => setMargin(Number(e.target.value) || 0)}
                    className="font-mono"
                  />
                </div>
              </div>

              {paperPreset === "custom" && (
                <div className="space-y-1">
                  <Label>Custom Dimensions (WIDTHxHEIGHT mm)</Label>
                  <Input
                    value={customPaper}
                    onChange={(e) => setCustomPaper(e.target.value)}
                    placeholder="250x250"
                    className="font-mono"
                  />
                </div>
              )}

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label>Hatch Spacing (mm)</Label>
                  <Input
                    type="number"
                    min={0}
                    step={0.1}
                    value={hatchSpacing}
                    onChange={(e) => setHatchSpacing(Number(e.target.value) || 0)}
                    className="font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Hatch Angle (deg)</Label>
                  <Input
                    type="number"
                    step={5}
                    value={hatchAngle}
                    onChange={(e) => setHatchAngle(Number(e.target.value) || 0)}
                    className="font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label>Draw Feed (mm/min)</Label>
                  <Input
                    type="number"
                    min={100}
                    max={6000}
                    step={100}
                    value={feed}
                    onChange={(e) => setFeed(Number(e.target.value) || 1000)}
                    className="font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Pen Lift Z (mm)</Label>
                  <Input
                    type="number"
                    min={0.5}
                    step={0.5}
                    value={lift}
                    onChange={(e) => setLift(Number(e.target.value) || 2)}
                    className="font-mono"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between pt-1">
                <Label htmlFor="outline-switch">Outline filled shapes</Label>
                <Switch
                  id="outline-switch"
                  checked={outline}
                  onCheckedChange={setOutline}
                />
              </div>

              <Button
                className="w-full"
                onClick={() => void runPipeline(rawBytes, sourceKind, pens)}
                disabled={busy !== null}
              >
                <RefreshCw className={busy ? "animate-spin" : ""} />
                {busy ?? "Compile SVG & G-code"}
              </Button>
            </CardContent>
          </Card>
        </aside>

        {/* Center Column: Interactive Paper Drafting Bed */}
        <main className="drafting-grid flex flex-col min-h-[540px] relative">
          <Tabs
            value={viewTab}
            onValueChange={setViewTab}
            className="flex-1 flex flex-col"
          >
            <div className="h-10 border-b border-[#DCD9D0] bg-[#FBFBF9]/90 backdrop-blur-xs px-4 flex items-center justify-between">
              <TabsList>
                <TabsTrigger value="toolpath" className="gap-1.5">
                  <Layers className="h-3.5 w-3.5" />
                  G-code Toolpath
                </TabsTrigger>
                <TabsTrigger value="svg" className="gap-1.5">
                  <ImageIcon className="h-3.5 w-3.5" />
                  Traced SVG Layers
                </TabsTrigger>
                <TabsTrigger value="raw" className="gap-1.5">
                  <FileCode2 className="h-3.5 w-3.5" />
                  Raw G-code ({gcodeText ? gcodeText.split("\n").length : 0} lines)
                </TabsTrigger>
              </TabsList>

              {viewTab === "toolpath" && (
                <div className="flex items-center gap-3">
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
                      Show rapid travel (G0)
                    </Label>
                  </div>
                </div>
              )}
            </div>

            {errorMessage && (
              <div className="mx-4 mt-3 rounded-sm border border-[#D52B1E] bg-[#D52B1E]/10 px-3 py-2 text-xs text-[#D52B1E] font-mono">
                {errorMessage}
              </div>
            )}

            {/* Toolpath Canvas */}
            <TabsContent
              value="toolpath"
              className="flex-1 flex flex-col items-center justify-center p-6 mt-0"
            >
              <div className="relative flex items-center justify-center w-full h-full max-h-[680px]">
                <svg
                  viewBox={`0 0 ${paperWidthMm} ${paperHeightMm}`}
                  className="max-w-full max-h-[640px] w-auto h-auto bg-[#FBFBF9] shadow-sm border border-[#231F20]/40"
                  style={{
                    aspectRatio: `${paperWidthMm} / ${paperHeightMm}`,
                  }}
                >
                  {/* Margin Guide Box */}
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

                  {/* Origin Crosshair (Bottom-Left in Paper Coordinates) */}
                  <g stroke="#D52B1E" strokeWidth={Math.max(paperWidthMm, paperHeightMm) * 0.003}>
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

                  {/* G-code Strokes (Paper Y is up, SVG Y is down) */}
                  {visibleSegments.map((seg, idx) => {
                    const penColor =
                      (seg.tool !== null &&
                        parsedGcode.tools.get(seg.tool)?.hex) ||
                      "#231F20"
                    return (
                      <line
                        key={idx}
                        x1={seg.x0}
                        y1={paperHeightMm - seg.y0}
                        x2={seg.x1}
                        y2={paperHeightMm - seg.y1}
                        stroke={seg.rapid ? "#9E9890" : penColor}
                        strokeWidth={
                          seg.rapid
                            ? Math.max(paperWidthMm, paperHeightMm) * 0.0018
                            : Math.max(paperWidthMm, paperHeightMm) * 0.004
                        }
                        strokeDasharray={seg.rapid ? "2 2" : undefined}
                        strokeLinecap="round"
                        opacity={seg.rapid ? 0.45 : 0.92}
                      />
                    )
                  })}
                </svg>
              </div>

              {/* Pen Tool Filter Bar */}
              <div className="mt-3 flex flex-wrap items-center gap-1.5 bg-[#FBFBF9] border border-[#DCD9D0] px-2.5 py-1.5 rounded-sm">
                <span className="text-[11px] text-[#6B6661] mr-1">
                  Filter tool:
                </span>
                <button
                  type="button"
                  onClick={() => setActiveToolFilter(null)}
                  className={`px-2 py-0.5 text-[11px] font-mono rounded-[2px] cursor-pointer ${
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
                    className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-[11px] font-mono rounded-[2px] border cursor-pointer ${
                      activeToolFilter === tool.tool
                        ? "border-[#231F20] bg-[#231F20] text-[#FBFBF9]"
                        : "border-[#DCD9D0] bg-[#FBFBF9] text-[#231F20]"
                    }`}
                  >
                    <span
                      className="h-2.5 w-2.5 rounded-full border border-black/20"
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
              className="flex-1 flex items-center justify-center p-6 mt-0"
            >
              <div
                className="max-w-full max-h-[640px] bg-[#FBFBF9] p-4 border border-[#DCD9D0] shadow-xs overflow-auto [&>svg]:max-h-[580px] [&>svg]:w-auto"
                dangerouslySetInnerHTML={{ __html: tracedSvg }}
              />
            </TabsContent>

            {/* Raw G-code View */}
            <TabsContent value="raw" className="flex-1 p-6 mt-0">
              <pre className="w-full h-[620px] overflow-auto rounded-sm border border-[#DCD9D0] bg-[#FBFBF9] p-3 text-[11px] font-mono leading-relaxed text-[#231F20]">
                {gcodeText || "; No G-code compiled yet"}
              </pre>
            </TabsContent>
          </Tabs>
        </main>

        {/* Right Column: Telemetry, Motion Summary & Owl 6.9 Arm Execution */}
        <aside className="border-l border-[#DCD9D0] bg-[#FBFBF9] p-4 space-y-4 overflow-y-auto">
          {/* Motion Summary */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle>Planned Motion Telemetry</CardTitle>
                <Sparkles className="h-3.5 w-3.5 text-[#6B6661]" />
              </div>
            </CardHeader>
            <CardContent className="space-y-2.5">
              {drawSummary ? (
                <>
                  <div className="grid grid-cols-3 gap-2">
                    <div className="rounded-sm bg-[#F2EFE9] p-2 border border-[#DCD9D0]">
                      <div className="text-[10px] text-[#6B6661]">Moves</div>
                      <div className="text-sm font-mono font-medium">
                        {drawSummary.moves.toLocaleString()}
                      </div>
                    </div>
                    <div className="rounded-sm bg-[#F2EFE9] p-2 border border-[#DCD9D0]">
                      <div className="text-[10px] text-[#6B6661]">Swaps</div>
                      <div className="text-sm font-mono font-medium">
                        {drawSummary.pen_changes}
                      </div>
                    </div>
                    <div className="rounded-sm bg-[#F2EFE9] p-2 border border-[#DCD9D0]">
                      <div className="text-[10px] text-[#6B6661]">Duration</div>
                      <div className="text-sm font-mono font-medium">
                        {(drawSummary.estimated_seconds / 60).toFixed(1)} min
                      </div>
                    </div>
                  </div>

                  <div className="rounded-sm bg-[#F2EFE9] p-2.5 border border-[#DCD9D0] space-y-1 font-mono text-[11px]">
                    <div className="text-[10px] font-sans font-medium text-[#6B6661] mb-1">
                      Base-Frame Pen-Tip Envelope (mm)
                    </div>
                    {["X", "Y", "Z"].map((axis, i) => (
                      <div key={axis} className="flex justify-between">
                        <span className="text-[#6B6661]">{axis} axis:</span>
                        <span>
                          {drawSummary.bounds_mm.min[i].toFixed(1)} ..{" "}
                          {drawSummary.bounds_mm.max[i].toFixed(1)} mm
                        </span>
                      </div>
                    ))}
                  </div>

                  {drawSummary.worst_error_mm !== undefined && (
                    <div className="rounded-sm border border-[#00843D] bg-[#00843D]/10 px-2.5 py-1.5 text-xs font-mono text-[#00843D]">
                      Completed · Peak tracking error:{" "}
                      {drawSummary.worst_error_mm.toFixed(2)} mm
                    </div>
                  )}
                </>
              ) : (
                <p className="text-xs text-[#6B6661]">
                  Compile a drawing to inspect base-frame bounds and duration.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Owl 6.9 Arm Placement & Execution */}
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle>Owl 6.9 Arm Placement</CardTitle>
                <button
                  type="button"
                  onClick={() => void handleTeachOrigin()}
                  disabled={busy !== null}
                  className="text-[11px] text-[#006699] hover:underline cursor-pointer disabled:opacity-50"
                >
                  Teach from arm
                </button>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1">
                <Label>Paper Bottom-Left Origin (Base X, Y, Z mm)</Label>
                <div className="grid grid-cols-3 gap-1.5">
                  <Input
                    value={originX}
                    onChange={(e) => setOriginX(e.target.value)}
                    placeholder="X"
                    className="font-mono"
                  />
                  <Input
                    value={originY}
                    onChange={(e) => setOriginY(e.target.value)}
                    placeholder="Y"
                    className="font-mono"
                  />
                  <Input
                    value={originZ}
                    onChange={(e) => setOriginZ(e.target.value)}
                    placeholder="Z"
                    className="font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1">
                  <Label>Pen Offset (mm)</Label>
                  <Input
                    value={penOffset}
                    onChange={(e) => setPenOffset(e.target.value)}
                    className="font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Trailing Tilt (deg)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={30}
                    value={tilt}
                    onChange={(e) => setTilt(Number(e.target.value) || 0)}
                    className="font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="space-y-1">
                  <Label>Rapid (mm/min)</Label>
                  <Input
                    type="number"
                    min={500}
                    max={6000}
                    step={250}
                    value={rapid}
                    onChange={(e) => setRapid(Number(e.target.value) || 3000)}
                    className="font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <Label>Swap Wait (s)</Label>
                  <Input
                    type="number"
                    min={0}
                    step={1}
                    value={swapWait}
                    onChange={(e) => setSwapWait(Number(e.target.value) || 0)}
                    className="font-mono"
                  />
                </div>
                <div className="space-y-1">
                  <Label>ZMQ Host</Label>
                  <Input
                    value={robotHost}
                    onChange={(e) => setRobotHost(e.target.value)}
                    className="font-mono"
                  />
                </div>
              </div>

              <div className="pt-1 space-y-2">
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
                    ? `Draw Only T${activeToolFilter} on Owl 6.9`
                    : "Draw on Owl 6.9 Arm"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </aside>
      </div>
    </div>
  )
}

export default App
