#!/usr/bin/env python3
"""Static file server with COOP/COEP headers, stdlib only.

Cross-origin isolation (`self.crossOriginIsolated === true`) is required for
`SharedArrayBuffer` and atomics, which the lean engine's threaded-wasm CPU
rung (`wasm-mt` feature, `wasm-bindgen-rayon`) needs to spin up its worker
pool. GitHub Pages and trucs.ai cannot set these headers directly - this
script is for local testing only (the gate this crate's threads work needs
to pass before shipping the loader's rung-selection logic), matching t0-web's
own local-only `coi_server.py` approach (idle-intelligence/t0-web PR #8).

Serves `_site` by default - run `scripts/build.sh` first - so the pages see
the same assembled tree GitHub Pages deploys.

Usage:
    python3 scripts/serve.py [--dir DIR] [--port PORT]
"""
import argparse
import http.server
import os
import socketserver
import sys


class CoiHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="_site")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    handler = lambda *a, **kw: CoiHandler(*a, directory=root, **kw)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {root} on http://{args.bind}:{args.port} (COOP/COEP set, Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
