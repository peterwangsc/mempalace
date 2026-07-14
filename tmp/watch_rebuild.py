#!/usr/bin/env python3.13
"""Live progress tracker for the mempalace HNSW rebuild.

Usage:  python3.13 tmp/watch_rebuild.py
Reads chroma.sqlite3 in read-only mode every 5s — safe alongside the rebuild.
Ctrl-C to quit (does not affect the rebuild).
"""

import sqlite3
import time
import sys

DB = "file:/Users/peterwang/.mempalace/palace/chroma.sqlite3?mode=ro"
TARGET = 183_069  # sqlite drawer count at rebuild start, 2026-07-14 01:18
BAR_WIDTH = 40


def rebuilt_count():
    conn = sqlite3.connect(DB, uri=True, timeout=2.0)
    try:
        row = conn.execute(
            """
            SELECT count(*)
            FROM embeddings e
            JOIN segments s ON e.segment_id = s.id
            JOIN collections c ON s.collection = c.id
            WHERE c.name = 'mempalace_drawers'
              AND s.scope = 'METADATA'
            """
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def fmt_eta(seconds):
    if seconds is None or seconds <= 0 or seconds != seconds:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    return f"{m:d}m{s:02d}s"


def main():
    samples = []  # (time, count)
    print(f"Rebuild target: {TARGET:,} drawers  (polling every 5s, Ctrl-C to quit)\n")
    while True:
        try:
            n = rebuilt_count()
        except sqlite3.Error as e:
            sys.stdout.write(f"\r  db busy ({e}) — retrying...".ljust(90))
            sys.stdout.flush()
            time.sleep(5)
            continue

        now = time.time()
        samples.append((now, n))
        samples[:] = samples[-24:]  # ~2 min rate window

        rate = None
        if len(samples) >= 2:
            dt = samples[-1][0] - samples[0][0]
            dn = samples[-1][1] - samples[0][1]
            rate = dn / dt if dt > 0 and dn > 0 else None

        pct = min(n / TARGET, 1.0)
        filled = int(BAR_WIDTH * pct)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        rate_s = f"{rate * 60:,.0f}/min" if rate else "--"
        eta = fmt_eta((TARGET - n) / rate) if rate else "--:--"
        line = f"\r  [{bar}] {n:>7,}/{TARGET:,} ({pct:5.1%})  {rate_s:>10}  ETA {eta:>7}"
        sys.stdout.write(line.ljust(95))
        sys.stdout.flush()

        if n >= TARGET:
            print("\n\n  Upsert phase complete — the rebuild's final verify/swap may run a bit longer.")
            print("  Check: tail -5 tmp/rebuild-20260714.log  (look for 'Repair complete')")
            break
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n(tracker stopped; rebuild unaffected)")
