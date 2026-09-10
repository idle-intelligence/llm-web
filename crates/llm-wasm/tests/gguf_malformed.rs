//! Malformed/adversarial GGUF header tests (no GPU) — every case here must
//! return `Err` from `GgufReader::open`, never panic or abort (OOM from an
//! unchecked `with_capacity`, index-out-of-bounds from an unchecked
//! offset/length, etc.). See docs/ENGINE.md's "Review fixes 2026-09-10"
//! list, finding 1/6/8.

use std::io::Cursor;

use llm_wasm::gguf::GgufReader;

/// Minimal hand-rolled GGUF byte-string writer — just enough of the format
/// (magic, version, counts, KV pairs, tensor infos) to build the malformed
/// fixtures below. Values are always little-endian, matching the GGUF spec.
#[derive(Default)]
struct GgufWriter {
    buf: Vec<u8>,
}

impl GgufWriter {
    fn new() -> Self {
        Self::default()
    }

    fn u32(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn u64(&mut self, v: u64) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    fn string(&mut self, s: &str) -> &mut Self {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
        self
    }

    /// magic + version, nothing else.
    fn header(version: u32) -> Self {
        let mut w = Self::new();
        w.buf.extend_from_slice(b"GGUF");
        w.u32(version);
        w
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// A single F32 tensor info descriptor: name, ndims, dims, dtype=F32(0), offset.
fn write_f32_tensor_info(w: &mut GgufWriter, name: &str, dims: &[u64], offset: u64) {
    w.string(name);
    w.u32(dims.len() as u32);
    for &d in dims {
        w.u64(d);
    }
    w.u32(0); // GgmlDtype::F32
    w.u64(offset);
}

#[test]
fn truncated_header_errors_not_panics() {
    // Magic + version only — cut off before tensor_count/metadata_kv_count
    // are even read.
    let bytes = GgufWriter::header(3).into_bytes();
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(result.is_err(), "truncated header must error, not panic");
}

#[test]
fn truncated_header_mid_magic_errors() {
    // Not even a full magic number.
    let bytes = vec![b'G', b'G'];
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(result.is_err());
}

#[test]
fn absurd_tensor_count_errors_not_oom() {
    // tensor_count claims u32::MAX-scale tensors but the file has no
    // tensor-info bytes to back that up. Must error on the first read
    // failure inside the tensor-index loop, not attempt a huge allocation
    // (the `with_capacity` hints are capped — see gguf.rs's
    // `MAX_CAPACITY_HINT`) and not panic.
    let mut w = GgufWriter::header(3);
    w.u64(u64::MAX / 2); // tensor_count
    w.u64(0); // metadata_kv_count
    let bytes = w.into_bytes();
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(result.is_err(), "absurd tensor_count must error, not OOM/panic");
}

#[test]
fn absurd_metadata_kv_count_errors_not_oom() {
    let mut w = GgufWriter::header(3);
    w.u64(0); // tensor_count
    w.u64(u64::MAX / 2); // metadata_kv_count
    let bytes = w.into_bytes();
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(
        result.is_err(),
        "absurd metadata_kv_count must error, not OOM/panic"
    );
}

#[test]
fn ndims_exceeding_max_errors() {
    // One tensor, well-formed except ndims=9 (> the supported max of 4).
    let mut w = GgufWriter::header(3);
    w.u64(1); // tensor_count
    w.u64(0); // metadata_kv_count
    w.string("bad_tensor");
    w.u32(9); // ndims — exceeds supported maximum
    for _ in 0..9 {
        w.u64(1);
    }
    w.u32(0); // dtype F32
    w.u64(0); // offset
    let bytes = w.into_bytes();
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(result.is_err(), "ndims=9 must be rejected, not read OOB");
}

#[test]
fn tensor_offset_plus_size_exceeding_file_errors() {
    // One well-formed tensor descriptor (ndims=1, dim=[1<<20], F32 = 4MB),
    // but the file is tiny — nowhere near big enough to actually hold that
    // tensor's data. `open()` must catch this before any caller can slice
    // past the end of the file.
    let mut w = GgufWriter::header(3);
    w.u64(1); // tensor_count
    w.u64(0); // metadata_kv_count
    write_f32_tensor_info(&mut w, "huge_tensor", &[1 << 20], 0);
    let bytes = w.into_bytes(); // no data section at all follows
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(
        result.is_err(),
        "tensor data range past EOF must be rejected at open() time"
    );
}

#[test]
fn tensor_offset_alone_past_eof_errors() {
    // Small tensor (harmless size) but an absurd per-tensor `offset` that
    // alone already exceeds the file length.
    let mut w = GgufWriter::header(3);
    w.u64(1); // tensor_count
    w.u64(0); // metadata_kv_count
    write_f32_tensor_info(&mut w, "small_tensor", &[4], u64::MAX / 2);
    let bytes = w.into_bytes();
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(result.is_err(), "offset past EOF must be rejected");
}

#[test]
fn well_formed_minimal_gguf_opens_successfully() {
    // Sanity check that the malformed-file rejections above aren't
    // accidentally rejecting valid files too: one small F32 tensor, data
    // section present and correctly sized.
    let mut w = GgufWriter::header(3);
    w.u64(1); // tensor_count
    w.u64(0); // metadata_kv_count
    write_f32_tensor_info(&mut w, "ok_tensor", &[4], 0);
    let mut bytes = w.into_bytes();
    // Pad to 32-byte alignment (ALIGNMENT in gguf.rs) then append the
    // tensor's 16 bytes (4 x f32) of data.
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    bytes.extend_from_slice(&[0u8; 16]);
    let result = GgufReader::open(Cursor::new(bytes));
    assert!(result.is_ok(), "well-formed minimal GGUF must still open: {:?}", result.err());
}
