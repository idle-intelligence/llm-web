// Copied verbatim from trucs.ai lib/model-cache.js
// (github.com/idle-intelligence/trucs.ai, branch llm-life-demo,
// commit 920fbd93a774595dda2b15ae50cc1d0e06c34ef6). Do not fork this file —
// pull a fresh copy from that branch instead of hand-editing it here.
// model-cache.js — shared model download + Cache API storage for any
// trucs.ai demo that loads one or more model files, in a page or a Worker.
// No dependencies, dependency-free ES module.
//
// Written after a survey of ~20 existing model download/caching
// implementations across trucs.ai and the idle-intelligence repos found no
// two of them sharing code. The versioned-cache-name-with-eviction pattern
// is copied from
// astres/terrain-cache.js (github.com/idle-intelligence/trucs.ai,
// origin/main) — `CACHE_NAME` + "evict caches from previous versions" —
// generalized here to any prefix-vN cache name instead of one hardcoded
// prefix.
//
// NOT supported, by decision: resuming partial downloads, HTTP revision
// pinning, OPFS, splitting one file into pieces across cache entries (a
// consumer that needs to feed an engine in fixed-size blocks, e.g. a GGUF
// shard reader, does that itself after getModel hands back the whole
// file's bytes — that's an engine-loading concern, not a caching one).

// Evict any other cache whose name shares `cacheName`'s "<prefix>-vN"
// convention but has a different version, so re-bumping the version number
// doesn't leave the old files on disk forever. A name with no "-vN" suffix
// is left alone (nothing to evict against).
async function evictOldVersions(cacheName) {
  const m = cacheName.match(/^(.*)-v\d+$/);
  if (!m) return;
  const prefix = m[1] + '-';
  try {
    const keys = await caches.keys();
    await Promise.all(
      keys.filter((k) => k.startsWith(prefix) && k !== cacheName).map((k) => caches.delete(k))
    );
  } catch (_) {
    // Best effort: eviction failing is not a reason to fail the load.
  }
}

async function estimateMb() {
  try {
    const { quota, usage } = await navigator.storage.estimate();
    return { quotaMb: Math.round(quota / 1e6), usageMb: Math.round(usage / 1e6) };
  } catch (_) {
    return null;
  }
}

async function fetchBytes(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url}: ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onBytes(loaded, total);
  }
  const n = chunks.reduce((s, c) => s + c.length, 0);
  const bytes = new Uint8Array(n);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  return bytes;
}

// Streams `url` straight into `cache`: one branch of a teed response body
// goes to cache.put (the browser writes it to disk without JS ever holding
// the whole file), the other branch is only counted for onBytes and its
// chunks are discarded. Reads the bytes back from the cache once put
// resolves, so peak JS memory holds one copy of the file instead of two
// (a 430 MB model used to mean a 430 MB chunk buffer, a second 430 MB
// concatenated copy, and a third copy inside cache.put's own Response —
// over 1 GB peak on a phone). Throws if the body can't be teed or the put
// fails; the caller falls back to plain fetchBytes.
async function fetchToCache(url, cache, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url}: ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  const [forCache, forCount] = r.body.tee();

  const putPromise = cache.put(
    url,
    new Response(forCache, { headers: { 'Content-Type': 'application/octet-stream' } })
  );

  const countPromise = (async () => {
    const reader = forCount.getReader();
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.length;
      onBytes(loaded, total);
    }
  })();

  await Promise.all([putPromise, countPromise]);

  const cached = await cache.match(url);
  return new Uint8Array(await cached.arrayBuffer());
}

// Asked at most once per page/worker lifetime (navigator.storage.persist()
// itself is idempotent, but there's no reason to call it more than once).
let persistAsked = false;

// getModel(urls, { cache, onProgress, onFile, onStatus })
//
// - urls: the model's files as they exist on the server, in order (a
//   single-file model is a list of one; a multi-shard model's list has one
//   entry per shard).
// - cache: a versioned cache name, e.g. 'qwen2.5-0.5b-instruct-gguf-v1'.
//   Bump the version when the files at those URLs change; the old version's
//   cache is evicted automatically. Omit to skip caching (network only).
// - onProgress(loaded, total): bytes loaded vs. total, summed across every
//   file in this call. `total` only includes files whose Content-Length (or
//   cached size) is known so far, so it can grow as later files start.
// - onFile(i, bytes): called as each file's bytes become available, in
//   `urls` order, so the caller can hand files to an engine one at a time
//   instead of waiting for the whole list.
// - onStatus(message): called at most once, only if storage failed (cache
//   open or a cache.put), with one plain sentence naming the reason and the
//   current storage quota/usage. The downloaded bytes are still returned
//   and handed to onFile either way — a storage failure never breaks the
//   caller, it just means a re-download next time.
//
// Returns a Promise of the files' bytes (Uint8Array[]), in `urls` order.
export async function getModel(urls, opts = {}) {
  const { cache: cacheName, onProgress = () => {}, onFile = () => {}, onStatus = () => {} } = opts;

  let cache = null;
  let storageFailure = null;
  if (cacheName) {
    try {
      cache = await caches.open(cacheName);
      await evictOldVersions(cacheName);
    } catch (err) {
      cache = null;
      storageFailure = err;
    }
  }

  const cached = new Array(urls.length).fill(null);
  if (cache) {
    for (let i = 0; i < urls.length; i++) {
      try { cached[i] = await cache.match(urls[i]); } catch (_) { /* treat as a miss */ }
    }
  }

  if (cached.some((r) => !r) && !persistAsked) {
    persistAsked = true;
    try { await navigator.storage.persist(); } catch (_) { /* best effort */ }
  }

  const sizes = new Array(urls.length).fill(0);
  const loaded = new Array(urls.length).fill(0);
  const report = () => onProgress(loaded.reduce((a, b) => a + b, 0), sizes.reduce((a, b) => a + b, 0));

  const results = [];
  for (let i = 0; i < urls.length; i++) {
    if (cached[i]) {
      const bytes = new Uint8Array(await cached[i].arrayBuffer());
      sizes[i] = bytes.length;
      loaded[i] = bytes.length;
      report();
      onFile(i, bytes);
      results.push(bytes);
      continue;
    }

    const onBytes = (l, t) => {
      loaded[i] = l;
      sizes[i] = t || sizes[i];
      report();
    };

    let bytes;
    if (cache) {
      try {
        bytes = await fetchToCache(urls[i], cache, onBytes);
      } catch (err) {
        if (!storageFailure) storageFailure = err;
        bytes = await fetchBytes(urls[i], onBytes);
      }
    } else {
      bytes = await fetchBytes(urls[i], onBytes);
    }
    sizes[i] = bytes.length;

    onFile(i, bytes);
    results.push(bytes);
  }

  if (storageFailure) {
    const est = await estimateMb();
    const reason = storageFailure.name || String(storageFailure);
    const quotaText = est ? `, ${est.usageMb} of ${est.quotaMb} MB used` : '';
    onStatus(`couldn't store the model (${reason}${quotaText}); it will download again next time`);
  }

  return results;
}
