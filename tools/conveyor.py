"""Host-side controller for the ESP32 conveyor firmware.

Usage:
    python tools/conveyor.py                     interactive prompt
    python tools/conveyor.py start               single command
    python tools/conveyor.py fwd start           several commands in order
    python tools/conveyor.py --port /dev/cu.usbserial-0001 status
    python tools/conveyor.py --test              exercise every command

Commands: start, stop, fwd, rev, status (case-insensitive).
Requires pyserial (pip install pyserial).
"""

import argparse
import sys
import time

import serial
from serial.tools import list_ports

DEFAULT_BAUD = 115200
REPLY_TIMEOUT_S = 1.0
COMMANDS = ("START", "STOP", "FWD", "REV", "STATUS")

# USB-UART bridges commonly found on ESP32 boards: CP210x, CH340/CH9102, FTDI, Espressif native USB.
KNOWN_VIDS = {0x10C4, 0x1A86, 0x0403, 0x303A}


def find_port():
    candidates = [p.device for p in list_ports.comports() if p.vid in KNOWN_VIDS]
    if not candidates:
        raise RuntimeError("No ESP32 serial port found; pass --port explicitly.")
    if len(candidates) > 1:
        print(f"Multiple ports found {candidates}, using {candidates[0]}", file=sys.stderr)
    return candidates[0]


class Conveyor:
    def __init__(self, port=None, baud=DEFAULT_BAUD):
        self.ser = serial.Serial()
        self.ser.port = port or find_port()
        self.ser.baudrate = baud
        self.ser.timeout = 0.1
        # The ESP32 auto-reset circuit pulls EN low when RTS is asserted and DTR is not.
        # Open with both asserted, then drop RTS before DTR so that state never occurs.
        self.ser.open()
        self.ser.rts = False
        self.ser.dtr = False
        time.sleep(0.05)
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def send(self, cmd):
        """Send a command and return (ok, fields) parsed from the reply line.

        A successful reply looks like "OK RUNNING FWD" -> (True, ["RUNNING", "FWD"]).
        An error reply looks like "ERR ..." -> (False, ["..."]).
        """
        self.ser.write((cmd.strip() + "\n").encode("ascii"))
        deadline = time.monotonic() + REPLY_TIMEOUT_S
        while time.monotonic() < deadline:
            line = self.ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("OK"):
                return True, line.split()[1:]
            if line.startswith("ERR"):
                return False, [line[3:].strip()]
        raise TimeoutError(f"No reply to '{cmd}' within {REPLY_TIMEOUT_S} s")

    def start(self):
        return self.send("START")

    def stop(self):
        return self.send("STOP")

    def forward(self):
        return self.send("FWD")

    def reverse(self):
        return self.send("REV")

    def status(self):
        return self.send("STATUS")


def run_command(conv, cmd):
    ok, fields = conv.send(cmd)
    print(("OK " if ok else "ERR ") + " ".join(fields))
    return ok


def interactive(conv):
    print(f"Connected to {conv.ser.port}. Commands: {', '.join(COMMANDS)}. Ctrl-D or 'quit' to exit.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        run_command(conv, line)


def self_test(conv):
    """Runs the motor briefly in both directions and checks every reply."""
    steps = [
        ("STOP",   ["STOPPED", None]),
        ("FWD",    ["STOPPED", "FWD"]),
        ("START",  ["RUNNING", "FWD"]),
        ("REV",    ["RUNNING", "REV"]),
        ("STOP",   ["STOPPED", "REV"]),
        ("FWD",    ["STOPPED", "FWD"]),
        ("STATUS", ["STOPPED", "FWD"]),
    ]
    failures = 0
    try:
        for cmd, expected in steps:
            ok, fields = conv.send(cmd)
            match = ok and all(e is None or e == f for e, f in zip(expected, fields))
            failures += not match
            print(f"{cmd:<7} -> {' '.join(fields):<16} {'pass' if match else 'FAIL'}")
            if cmd in ("START", "REV"):
                time.sleep(1.0)
        ok, fields = conv.send("BOGUS")
        match = not ok
        failures += not match
        print(f"{'BOGUS':<7} -> ERR {' '.join(fields):<12} {'pass' if match else 'FAIL'}")
    finally:
        conv.stop()
    print(f"{len(steps) + 1 - failures}/{len(steps) + 1} passed")
    return failures == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control the ESP32 conveyor over UART.")
    parser.add_argument("commands", nargs="*", help="commands to send in order; omit for interactive mode")
    parser.add_argument("-p", "--port", help="serial port (auto-detected if omitted)")
    parser.add_argument("-b", "--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--test", action="store_true", help="run the motor through every command and verify replies")
    args = parser.parse_args()

    with Conveyor(args.port, args.baud) as conv:
        if args.test:
            sys.exit(0 if self_test(conv) else 1)
        if args.commands:
            sys.exit(0 if all(run_command(conv, c) for c in args.commands) else 1)
        interactive(conv)
