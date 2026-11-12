# ruff: noqa: INP001
"""Sample peak RSS or live Python blocks while a callable runs.

Resident set is sampled from another thread. NumPy releases the GIL inside
large copies and ``gemm``, and a ``SIGALRM`` handler on the main thread is
not delivered until that call returns, which is after the temporary has been
freed. ``ru_maxrss`` is not used: importing VTK already sets a high-water
mark above these workloads, so the kernel counter would not move.

Live Python blocks are sampled with ``SIGALRM``. ``sys.getallocatedblocks``
has to run on the allocating thread, and the string conversion does its
work in Python bytecode where the alarm is delivered.
"""

from __future__ import annotations

import gc
import pathlib
import signal
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def rss_bytes() -> int:
    """Return the process resident set size in bytes."""
    with pathlib.Path('/proc/self/status').open(encoding='ascii') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    message = 'VmRSS is missing from /proc/self/status'
    raise RuntimeError(message)


def peak_rss_delta(call: Callable[[], object]) -> int:
    """Return bytes of resident memory added at the high point of ``call``.

    The baseline is the resident set immediately before ``call``, after a
    collection. Inputs built before ``call`` are not part of the result.
    One fresh process per sample: freed pages stay in the resident set, so a
    second call in the same process under-reports the temporary.
    """
    gc.collect()
    before = rss_bytes()
    peak = before
    running = True

    def sample() -> None:
        nonlocal peak
        while running:
            current = rss_bytes()
            peak = max(peak, current)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        call()
    finally:
        running = False
        thread.join()
    return max(peak - before, 0)


def peak_allocation_delta(call: Callable[[], object]) -> int:
    """Return the peak increase in ``sys.getallocatedblocks`` during ``call``.

    This is the largest number of additional live Python blocks held at once,
    not a count of blocks that were freed before the peak.
    """
    gc.collect()
    before = sys.getallocatedblocks()
    peak = before

    def handler(_signum: int, _frame: object) -> None:
        nonlocal peak
        current = sys.getallocatedblocks()
        peak = max(peak, current)

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.0005, 0.0005)
    try:
        call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
    final = sys.getallocatedblocks()
    peak = max(peak, final)
    return max(peak - before, 0)
