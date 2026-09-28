#!/usr/bin/env python3
"""Loads a lean www/*mem_profile*.html page in Playwright's bundled headless
Chromium, waits for window.__leanResult, and reports it plus a coarse
Chromium-process RSS footprint (renderer + GPU helper) read via `ps`. Guards
against runaway memory: polls RSS in the background and kills its own
browser (never another process) if the total crosses --limit-gb.

Usage: python3 run_mem_profile.py <url> [--limit-gb 6] [--timeout 180]
"""
import argparse
import asyncio
import json
import subprocess
import sys

from playwright.async_api import async_playwright

CHROME_ARGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-angle=metal",
]


def rss_bytes_for_pids(pids):
    if not pids:
        return 0
    out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)], capture_output=True, text=True)
    total = 0
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            total += int(line) * 1024  # ps rss is in KiB
    return total


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--limit-gb", type=float, default=6.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    def chrome_pids():
        # `ps aux` (not `pgrep -f`, whose own argv contains the search
        # string and self-matches) filtered on the full command column, so
        # `ps`/`grep`'s own processes never show up. Playwright's
        # `headless=True` launches the "chrome-headless-shell" binary (a
        # separate executable from the full "Google Chrome for Testing.app"
        # used headed/by-channel), so both name forms are matched.
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        pids = set()
        for line in out.stdout.splitlines():
            if "chrome-headless-shell" in line or "Chrome for Testing" in line:
                parts = line.split()
                if len(parts) > 1:
                    pids.add(int(parts[1]))
        return pids

    pids_before = chrome_pids()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=CHROME_ARGS)
        page = await browser.new_page()
        logs = []
        page.on("console", lambda m: logs.append(m.text))

        killed = {"flag": False}
        peak = {"bytes": 0}

        async def guard():
            while not killed["flag"]:
                # Every Chrome-for-Testing pid that appeared after this
                # session's own launch (all of the browser's own process
                # tree: main + renderer + GPU helper) - never another
                # session's browser.
                all_pids = list(chrome_pids() - pids_before)
                total = rss_bytes_for_pids(all_pids)
                if total > peak["bytes"]:
                    peak["bytes"] = total
                if total > args.limit_gb * 1e9:
                    print(f"[guard] footprint {total/1e9:.2f}GB exceeded limit {args.limit_gb}GB - killing this session's browser only", file=sys.stderr)
                    killed["flag"] = True
                    await browser.close()
                    return
                await asyncio.sleep(0.25)

        guard_task = asyncio.create_task(guard())

        try:
            await page.goto(args.url, timeout=int(args.timeout * 1000))
            await page.wait_for_function("window.__leanResult !== undefined", timeout=args.timeout * 1000)
            result = await page.evaluate("window.__leanResult")
        finally:
            killed["flag"] = True
            await guard_task
            if browser.is_connected():
                all_pids = list(chrome_pids() - pids_before)
                final_rss = rss_bytes_for_pids(all_pids)
                if final_rss > peak["bytes"]:
                    peak["bytes"] = final_rss
                await browser.close()
            print(f"[mem] peak chromium process-tree RSS observed: {peak['bytes']/1e9:.3f}GB", file=sys.stderr)

        print("\n".join(logs))
        print("RESULT_JSON=" + json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
