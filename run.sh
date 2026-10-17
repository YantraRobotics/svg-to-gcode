#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PID_FILE=server.pid
LOG_FILE=server.log
STOP_TIMEOUT_S=10

running() {
  kill -0 -- "-$1" 2>/dev/null
}

stop_server() {
  [ -f "$PID_FILE" ] || return 0
  local pid
  pid="$(cat "$PID_FILE")"
  if running "$pid"; then
    echo "stopping the server (pid $pid)"
    # uv run passes SIGTERM on to the server, which stops as it does on Ctrl-C.
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq "$STOP_TIMEOUT_S"); do
      running "$pid" || break
      sleep 1
    done
    if running "$pid"; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
}

if [ "${1:-}" = "stop" ]; then
  stop_server
  exit 0
fi

PORT="${1:-8000}"
BIND="${2:-0.0.0.0}"

if [ ! -d web/node_modules ]; then
  npm --prefix web install
fi

npm --prefix web run build
stop_server

if ! command -v setsid >/dev/null; then
  echo "setsid not found; running in the foreground"
  exec uv run main.py --port "$PORT" --bind "$BIND"
fi

# setsid gives the server its own session, so closing SSH does not hang it up.
PYTHONUNBUFFERED=1 setsid uv run main.py --port "$PORT" --bind "$BIND" \
  >"$LOG_FILE" 2>&1 </dev/null &
echo $! >"$PID_FILE"
echo "server running in the background (pid $!), logging to $LOG_FILE"
echo "Ctrl-C only stops following the log; ./run.sh stop stops the server"
exec tail -n +1 -f "$LOG_FILE"
