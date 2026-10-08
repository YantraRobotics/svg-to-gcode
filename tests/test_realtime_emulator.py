import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import realtime_emulator

LAUNCHER = Path(realtime_emulator.__file__)
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="Mach scheduling policies are macOS-only"
)


def on_worker_thread(function, *args):
    # A scheduling policy sticks to its thread, so keep it off the test runner's.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(function, *args).result()


def test_the_tick_budget_is_granted():
    on_worker_thread(
        realtime_emulator.make_current_thread_realtime,
        realtime_emulator.TICK_MS,
        realtime_emulator.COMPUTATION_MS,
        realtime_emulator.TICK_MS,
    )


def test_a_budget_longer_than_its_deadline_is_refused():
    with pytest.raises(OSError, match="kern_return_t"):
        on_worker_thread(realtime_emulator.make_current_thread_realtime, 5.0, 10.0, 5.0)


def test_the_emulator_runs_as_main_with_its_arguments_and_siblings(tmp_path):
    (tmp_path / "sibling.py").write_text("VALUE = 'imported'\n")
    emulator = tmp_path / "emulator.py"
    emulator.write_text(
        "import sys\nimport sibling\nprint(__name__, sibling.VALUE, *sys.argv[1:])\n"
    )
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), str(emulator), "--mujoco", "controller"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["__main__", "imported", "--mujoco", "controller"]


@pytest.mark.parametrize("arguments", [[], ["missing.py"]])
def test_refuses_to_start_without_an_emulator(tmp_path, arguments):
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert result.stderr
