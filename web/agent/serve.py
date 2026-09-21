#!/usr/bin/env python3
"""Dev server for web/agent/, stdlib only.

Serves this directory (index.html, worker.js, llm-client.js, pkg/) with
Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy set so the page
can use WebGPU + a cross-origin Worker fetching model bytes from the
separate model server (scripts/serve_models.py, a different origin/port),
plus permissive CORS so this page's own fetches (of, say, the wasm-pack
`pkg/` files) aren't blocked either.

This is the third of three local dev servers documented in
docs/ENGINE.md's Browser section:
  1. scripts/serve_models.py --dir ~/Code/idle-intelligence/models  (port 8001, GGUF + tokenizer)
  2. (reserved — MCP tool server, out of scope here)
  3. web/agent/serve.py                                             (port 8002, this page)

Usage:
    python3 web/agent/serve.py [--port PORT] [--bind BIND]
"""
import argparse
import http.server
import os
import socketserver
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


class AgentPageHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Required for a page that wants WebGPU + SharedArrayBuffer-grade
        # isolation while its Worker cross-origin-fetches model bytes.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        # This page's own assets (pkg/*.wasm, pkg/*.js) served permissively;
        # the actual model bytes come from scripts/serve_models.py, which
        # sets its own CORS headers.
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()

    handler = lambda *a, **kw: AgentPageHandler(*a, directory=ROOT, **kw)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {ROOT} on http://{args.bind}:{args.port} (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
