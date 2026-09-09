#!/usr/bin/env python3
"""Static file server for GGUF model shards, stdlib only.

http.server's SimpleHTTPRequestHandler doesn't support HTTP Range
requests, which the browser needs for shard-by-shard model loading
(fetch() with a Range header, or partial re-fetch on retry). This adds
Range/If-Range handling, CORS, and Cross-Origin-Resource-Policy so the
page (served from a different origin) can fetch the weights.

Usage:
    python3 scripts/serve_models.py [--dir DIR] [--port PORT] [--bind BIND]
"""
import argparse
import email.utils
import http.server
import mimetypes
import os
import re
import socketserver
import sys

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), fmt % args)
        )

    def do_GET(self):
        self._serve(send_body=True)

    def do_HEAD(self):
        self._serve(send_body=False)

    def _serve(self, send_body):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return

        file_size = os.path.getsize(path)
        ctype = self.guess_type(path)
        last_modified = email.utils.formatdate(
            os.path.getmtime(path), usegmt=True
        )

        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        is_partial = False

        if range_header:
            m = RANGE_RE.match(range_header.strip())
            if not m:
                self.send_error(416, "Invalid Range header")
                self.send_header("Content-Range", f"bytes */{file_size}")
                return
            start_s, end_s = m.groups()
            if start_s == "" and end_s == "":
                self.send_error(416, "Invalid Range header")
                return
            if start_s == "":
                # suffix range: last N bytes
                suffix_len = int(end_s)
                start = max(file_size - suffix_len, 0)
                end = file_size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s != "" else file_size - 1
            if start >= file_size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            end = min(end, file_size - 1)
            is_partial = True

        length = end - start + 1

        self.send_response(206 if is_partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Last-Modified", last_modified)
        self.send_header("Accept-Ranges", "bytes")
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        if not send_body:
            return

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk_size = 1024 * 1024
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    mimetypes.add_type("application/octet-stream", ".gguf")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=os.path.expanduser("~/Code/idle-intelligence/models/gguf"),
        help="Directory to serve (default: ~/Code/idle-intelligence/models/gguf)",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()

    directory = os.path.abspath(os.path.expanduser(args.dir))
    if not os.path.isdir(directory):
        sys.exit(f"error: directory does not exist: {directory}")

    handler = lambda *a, **kw: RangeHTTPRequestHandler(*a, directory=directory, **kw)
    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {directory} on http://{args.bind}:{args.port} (Ctrl-C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
