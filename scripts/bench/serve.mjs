// Static server for the bench pages with a POST /report sink.
//   node serve.mjs <root dir> <port> <report dir>
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const [root, port, reportDir] = process.argv.slice(2);
const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  if (req.method === "POST" && url.pathname === "/report") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => { fs.mkdirSync(reportDir, { recursive: true }); fs.writeFileSync(path.join(reportDir, url.searchParams.get("name").replace(/[^\w.-]/g, "_") + ".json"), body); res.end("ok"); });
    return;
  }
  const f = path.join(root, decodeURIComponent(url.pathname));
  if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.statusCode = 404; return res.end("not found"); }
  res.setHeader("Content-Type", types[path.extname(f)] || "application/octet-stream");
  res.setHeader("Content-Length", fs.statSync(f).size);
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  fs.createReadStream(f).pipe(res);
}).listen(Number(port), "127.0.0.1");
