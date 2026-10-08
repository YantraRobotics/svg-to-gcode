"""Run the control repo's firmware emulator with its tick thread on macOS real-time scheduling.

The emulator ticks every 5 ms and the controller faults when feedback is over 25 ms old.
Ordinary macOS timeshare scheduling pauses a thread for 10-20 ms often enough to trip that,
so the tick thread needs the Mach time-constraint policy, the macOS counterpart of the
SCHED_FIFO priority the controller only requests on Linux.

usage: python realtime_emulator.py EMULATOR [EMULATOR_ARGS ...]
"""

import ctypes
import runpy
import sys
from pathlib import Path

THREAD_TIME_CONSTRAINT_POLICY = 2
NS_PER_MS = 1_000_000
TICK_MS = 5.0
# The kernel raises any computation below constraint / 2 to constraint / 2.
COMPUTATION_MS = TICK_MS / 2


class Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class TimeConstraintPolicy(ctypes.Structure):
    _fields_ = [
        ("period", ctypes.c_uint32),
        ("computation", ctypes.c_uint32),
        ("constraint", ctypes.c_uint32),
        ("preemptible", ctypes.c_int),
    ]


def make_current_thread_realtime(
    period_ms: float, computation_ms: float, constraint_ms: float
) -> None:
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libsystem.pthread_self.restype = ctypes.c_void_p
    libsystem.pthread_mach_thread_np.argtypes = [ctypes.c_void_p]
    libsystem.pthread_mach_thread_np.restype = ctypes.c_uint32
    libsystem.thread_policy_set.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]

    timebase = Timebase()
    libsystem.mach_timebase_info(ctypes.byref(timebase))

    def absolute_time(ms: float) -> int:
        return round(ms * NS_PER_MS * timebase.denom / timebase.numer)

    policy = TimeConstraintPolicy(
        absolute_time(period_ms),
        absolute_time(computation_ms),
        absolute_time(constraint_ms),
        1,
    )
    thread = libsystem.pthread_mach_thread_np(libsystem.pthread_self())
    words = ctypes.sizeof(policy) // ctypes.sizeof(ctypes.c_int)
    result = libsystem.thread_policy_set(
        thread, THREAD_TIME_CONSTRAINT_POLICY, ctypes.byref(policy), words
    )
    if result != 0:
        raise OSError(f"thread_policy_set failed with kern_return_t {result}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    emulator = Path(sys.argv[1]).resolve()
    if not emulator.is_file():
        print(f"error: no emulator script at {emulator}", file=sys.stderr)
        return 2
    make_current_thread_realtime(TICK_MS, COMPUTATION_MS, TICK_MS)
    sys.argv = sys.argv[1:]
    sys.path[0] = str(emulator.parent)
    runpy.run_path(str(emulator), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
