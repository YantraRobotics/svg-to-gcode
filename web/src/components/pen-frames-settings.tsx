import { useCallback, useEffect, useState } from "react"
import { Crosshair, RefreshCw, X } from "lucide-react"

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
import {
  DEFAULT_TOUCH_FRAMES,
  FRAME_POINTS_STORAGE_KEY,
  frameShift,
  listTargetFrames,
  PEN_FRAMES_STORAGE_KEY,
  REFERENCE_FRAME,
  type FramePoint,
  type FramePoints,
  type PenFrames,
  type TargetFrames,
} from "@/lib/pen-frames"

// Radix Select reserves the empty string, so "no frame" needs a value of its own.
const MANUAL_SWAP = "__manual__"

interface PenFramesSettingsProps {
  pens: { name: string; hex: string }[]
  frames: PenFrames
  points: FramePoints
  robotHost: string
  penOffset: string
  onSave: (frames: PenFrames, points: FramePoints) => void
  onClose: () => void
}

const EMPTY_POINT: FramePoint = ["", "", ""]

export function PenFramesSettings({
  pens,
  frames,
  points,
  robotHost,
  penOffset,
  onSave,
  onClose,
}: PenFramesSettingsProps) {
  const [draft, setDraft] = useState<PenFrames>(frames)
  const [pointsDraft, setPointsDraft] = useState<FramePoints>(points)
  const [served, setServed] = useState<TargetFrames | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [teaching, setTeaching] = useState<string | null>(null)
  const touchFrames = served?.eligible.length
    ? served.eligible
    : DEFAULT_TOUCH_FRAMES

  const setPointAxis = (frame: string, axis: 0 | 1 | 2, value: string) => {
    setPointsDraft((prev) => {
      const next: FramePoint = [...(prev[frame] ?? EMPTY_POINT)]
      next[axis] = value
      return { ...prev, [frame]: next }
    })
  }

  const teachPoint = async (frame: string) => {
    setTeaching(frame)
    setError(null)
    try {
      const params = new URLSearchParams({
        pen_offset: penOffset,
        host: robotHost,
        frame,
      })
      const res = await fetch(`/teach?${params.toString()}`, { method: "POST" })
      const payload = await res.json()
      if (!res.ok) {
        throw new Error(payload.error || "Teach failed")
      }
      const [x, y, z] = payload.origin as [number, number, number]
      setPointsDraft((prev) => ({
        ...prev,
        [frame]: [String(x), String(y), String(z)],
      }))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setTeaching(null)
    }
  }

  const fetchFrames = useCallback(
    () =>
      listTargetFrames(robotHost).then(
        (listed) => {
          setServed(listed)
          setError(null)
          setLoading(false)
        },
        (err: unknown) => {
          setServed(null)
          setError(err instanceof Error ? err.message : String(err))
          setLoading(false)
        }
      ),
    [robotHost]
  )

  useEffect(() => {
    void fetchFrames()
  }, [fetchFrames])

  const setFrame = (penName: string, frame: string) => {
    setDraft((prev) => {
      const next = { ...prev }
      if (frame && frame !== MANUAL_SWAP) {
        next[penName] = frame
      } else {
        delete next[penName]
      }
      return next
    })
  }

  const handleSave = () => {
    const cleaned = Object.fromEntries(
      Object.entries(draft).map(([name, frame]) => [name, frame.trim()])
    )
    try {
      window.localStorage.setItem(PEN_FRAMES_STORAGE_KEY, JSON.stringify(cleaned))
      window.localStorage.setItem(
        FRAME_POINTS_STORAGE_KEY,
        JSON.stringify(pointsDraft)
      )
    } catch {
      // the mapping still applies for this session
    }
    onSave(cleaned, pointsDraft)
    onClose()
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-[#231F20]/30"
      onClick={onClose}
    >
      <div
        className="w-[440px] rounded-sm border border-[#DCD9D0] bg-[#FBFBF9] shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-[#DCD9D0] px-4 py-2.5">
          <h2 className="text-xs font-semibold">Settings · Pen Target Frames</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-[#6B6661] hover:text-[#231F20] cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>

        <div className="space-y-3 px-4 py-3">
          <p className="text-[11px] text-[#6B6661]">
            Map each mounted pen to the controller target frame at its tip. The
            pen on the active frame draws first. After each pen the arm rises
            100 mm, switches to the next pen's frame and turns only W3 to bring
            its nib where the last one was. Once any pen has a frame, every pen
            a drawing uses needs one; with none mapped, pens are swapped by
            hand.
          </p>

          <div className="flex items-center justify-between rounded-sm border border-[#DCD9D0] bg-[#F2EFE9]/60 px-2.5 py-1.5 text-[11px] font-mono">
            <span className="text-[#6B6661]">
              {served
                ? `current: ${served.current} · ${served.eligible.length} frames`
                : error
                  ? "frames unavailable"
                  : "listing frames..."}
            </span>
            <button
              type="button"
              onClick={() => {
                setLoading(true)
                void fetchFrames()
              }}
              disabled={loading}
              className="inline-flex items-center gap-1 text-[#006699] hover:underline cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>
          {error && (
            <div className="rounded-sm border border-[#D52B1E] bg-[#D52B1E]/10 px-2.5 py-1.5 text-[11px] font-mono text-[#D52B1E]">
              {error}
            </div>
          )}

          <div className="space-y-1.5">
            <Label className="text-[11px] text-[#6B6661]">Pen → target frame</Label>
            {pens.map((pen) => {
              const frame = draft[pen.name] ?? ""
              const options = served
                ? Array.from(new Set([...served.eligible, frame].filter(Boolean)))
                : []
              return (
                <div
                  key={pen.name}
                  className="grid grid-cols-[14px_1fr_200px] items-center gap-2"
                >
                  <span
                    className="h-3 w-3 rounded-full border border-black/20"
                    style={{ backgroundColor: pen.hex }}
                  />
                  <span className="truncate text-xs font-mono">{pen.name}</span>
                  {served ? (
                    <Select
                      value={frame || MANUAL_SWAP}
                      onValueChange={(value) => setFrame(pen.name, value)}
                    >
                      <SelectTrigger className="h-7 font-mono text-[11px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value={MANUAL_SWAP}>Manual swap</SelectItem>
                        {options.map((option) => (
                          <SelectItem
                            key={option}
                            value={option}
                            className="font-mono"
                          >
                            {option}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Input
                      value={frame}
                      onChange={(e) => setFrame(pen.name, e.target.value)}
                      placeholder="target_frame_1"
                      className="h-7 font-mono text-[11px]"
                    />
                  )}
                </div>
              )
            })}
          </div>

          <div className="space-y-1.5">
            <Label className="text-[11px] text-[#6B6661]">
              Touch points X, Y, Z (mm)
            </Label>
            <p className="text-[11px] text-[#6B6661]">
              Set each frame active and touch its nib to the same physical
              point, then teach it. Corners are taught at {REFERENCE_FRAME}; the
              other frames draw shifted by how far their point reads from its.
              Leave all empty to draw unshifted.
            </p>
            {touchFrames.map((frame) => {
              const point = pointsDraft[frame] ?? EMPTY_POINT
              const shift =
                frame === REFERENCE_FRAME ? null : frameShift(pointsDraft, frame)
              return (
                <div key={frame} className="space-y-0.5">
                  <div className="grid grid-cols-[96px_1fr_1fr_1fr_24px] items-center gap-1">
                    <span className="truncate text-[11px] font-mono">
                      {frame}
                    </span>
                    {([0, 1, 2] as const).map((axis) => (
                      <Input
                        key={axis}
                        value={point[axis]}
                        onChange={(e) => setPointAxis(frame, axis, e.target.value)}
                        className="h-6 px-1.5 font-mono text-[11px]"
                      />
                    ))}
                    <button
                      type="button"
                      onClick={() => void teachPoint(frame)}
                      disabled={teaching !== null}
                      title={`Teach ${frame} from the arm (it must be the active frame)`}
                      className="h-6 w-6 inline-flex items-center justify-center rounded-sm border border-[#DCD9D0] bg-[#F2EFE9]/60 text-[#006699] hover:border-[#006699] cursor-pointer disabled:opacity-40"
                    >
                      <Crosshair
                        className={`h-3 w-3 ${teaching === frame ? "animate-pulse" : ""}`}
                      />
                    </button>
                  </div>
                  {shift && (
                    <div className="pl-[100px] text-[10px] font-mono text-[#6B6661]">
                      shift{" "}
                      {shift.map((value) => value.toFixed(2)).join(", ")} mm
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        <div className="flex justify-end gap-2 border-t border-[#DCD9D0] px-4 py-2.5">
          <Button variant="outline" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button size="sm" onClick={handleSave}>
            Save
          </Button>
        </div>
      </div>
    </div>
  )
}
