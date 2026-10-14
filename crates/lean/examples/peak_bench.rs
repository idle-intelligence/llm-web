//! Measured device ceilings for the roofline: memory read bandwidth, copy
//! bandwidth and F32 FMA throughput, timed with pass timestamps on a
//! wgpu device of its own. Fixed kernels, no tuning loop.
//!
//! `cargo run --release -p lean --example peak_bench [-- --mib 256 --reps 9]`

const READ_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * 256u;
    var acc = vec4<f32>(0.0);
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        acc += src[i];
        i += stride;
    }
    dst[gid.x] = acc.x + acc.y + acc.z + acc.w;
}
"#;

const COPY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let stride = nwg.x * 256u;
    var i = gid.x;
    loop {
        if (i >= p.n) { break; }
        dst[i] = src[i];
        i += stride;
    }
}
"#;

// 8 independent vec4 FMA chains, 2048 iterations: 8 * 4 * 2 * 2048 = 131072 FLOP per thread.
const FMA_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = f32(gid.x) * 1e-9;
    var a0 = vec4<f32>(s, s + 1.0, s + 2.0, s + 3.0);
    var a1 = a0 * 0.5; var a2 = a0 * 0.25; var a3 = a0 * 0.125;
    var a4 = a0 * 0.75; var a5 = a0 * 0.875; var a6 = a0 * 0.625; var a7 = a0 * 0.375;
    let m = vec4<f32>(0.9999, 0.9998, 0.9997, 0.9996);
    let c = vec4<f32>(0.0001, 0.0002, 0.0003, 0.0004);
    for (var i = 0u; i < 2048u; i++) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    let r = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    dst[gid.x] = r.x + r.y + r.z + r.w;
}
"#;

// Same FLOP count per thread as FMA_WGSL, as 16 independent scalar chains.
const FMA_SCALAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = f32(gid.x) * 1e-9;
    var a0 = s; var a1 = s + 1.0; var a2 = s + 2.0; var a3 = s + 3.0;
    var a4 = s + 4.0; var a5 = s + 5.0; var a6 = s + 6.0; var a7 = s + 7.0;
    var a8 = s + 8.0; var a9 = s + 9.0; var a10 = s + 10.0; var a11 = s + 11.0;
    var a12 = s + 12.0; var a13 = s + 13.0; var a14 = s + 14.0; var a15 = s + 15.0;
    let m = 0.9999;
    let c = 0.0001;
    for (var i = 0u; i < 4096u; i++) {
        a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
        a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
        a8 = fma(a8, m, c); a9 = fma(a9, m, c); a10 = fma(a10, m, c); a11 = fma(a11, m, c);
        a12 = fma(a12, m, c); a13 = fma(a13, m, c); a14 = fma(a14, m, c); a15 = fma(a15, m, c);
    }
    dst[gid.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}
"#;

fn arg(args: &[String], name: &str, default: u64) -> u64 {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.parse().unwrap()).unwrap_or(default)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mib = arg(&args, "--mib", 256);
    let reps = arg(&args, "--reps", 9) as usize;
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("adapter");
    assert!(adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY), "adapter lacks TIMESTAMP_QUERY");
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TIMESTAMP_QUERY,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    println!("adapter {} ({:?}, {:?})", info.name, info.backend, info.device_type);

    let bytes = mib << 20;
    let n4 = (bytes / 16) as u32;
    let mk = |size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false });
    let src = mk(bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
    let dst_copy = mk(bytes, wgpu::BufferUsages::STORAGE);
    let dst_small = mk(1 << 24, wgpu::BufferUsages::STORAGE);
    // Non-trivial contents so nothing is compressed away.
    let mut chunk = vec![0u8; 1 << 20];
    for (i, b) in chunk.iter_mut().enumerate() {
        *b = (i.wrapping_mul(2654435761) >> 7) as u8;
    }
    for off in (0..bytes).step_by(chunk.len()) {
        queue.write_buffer(&src, off, &chunk);
    }
    let params = mk(16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
    queue.write_buffer(&params, 0, bytemuck_u32(&[n4, 0, 0, 0]));

    let pipe = |src: &str| {
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &m, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
    };
    let p_read = pipe(READ_WGSL);
    let p_copy = pipe(COPY_WGSL);
    let p_fma = pipe(FMA_WGSL);
    let p_fma_s = pipe(FMA_SCALAR_WGSL);
    let bg = |p: &wgpu::ComputePipeline, e: &[(u32, &wgpu::Buffer)]| {
        let entries: Vec<wgpu::BindGroupEntry> = e.iter().map(|(b, buf)| wgpu::BindGroupEntry { binding: *b, resource: buf.as_entire_binding() }).collect();
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries })
    };
    let bg_read = bg(&p_read, &[(0, &src), (1, &dst_small), (2, &params)]);
    let bg_copy = bg(&p_copy, &[(0, &src), (1, &dst_copy), (2, &params)]);
    let bg_fma = bg(&p_fma, &[(0, &dst_small)]);
    let bg_fma_s = bg(&p_fma_s, &[(0, &dst_small)]);

    let qs = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 });
    let resolve = mk(16, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
    let staging = mk(16, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let period = f64::from(queue.get_timestamp_period());

    let time = |p: &wgpu::ComputePipeline, g: &wgpu::BindGroup, wgs: u32| -> f64 {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: &qs, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }),
            });
            pass.set_pipeline(p);
            pass.set_bind_group(0, g, &[]);
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        enc.resolve_query_set(&qs, 0..2, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, 16);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait).unwrap();
        let t = {
            let d = slice.get_mapped_range();
            let a = u64::from_le_bytes(d[0..8].try_into().unwrap());
            let b = u64::from_le_bytes(d[8..16].try_into().unwrap());
            (b - a) as f64 * period * 1e-9
        };
        staging.unmap();
        t
    };

    // Workgroup counts are fixed by the data size and the device limit, not searched.
    let wgs_stream = 4096u32;
    let mut rd = Vec::new();
    let mut cp = Vec::new();
    for r in 0..=reps {
        let a = time(&p_read, &bg_read, wgs_stream);
        let b = time(&p_copy, &bg_copy, wgs_stream);
        if r > 0 {
            rd.push(a);
            cp.push(b);
        }
    }
    let gb = bytes as f64 / 1e9;
    println!("read  {mib} MiB: median {:.3} ms = {:.1} GB/s (min {:.3} ms = {:.1} GB/s)", median(&mut rd) * 1e3, gb / median(&mut rd), rd[0] * 1e3, gb / rd[0]);
    println!("copy  {mib} MiB each way: median {:.3} ms = {:.1} GB/s moved (read+write)", median(&mut cp) * 1e3, 2.0 * gb / median(&mut cp));
    let wgs_fma = 1024u32;
    let flop = wgs_fma as f64 * 256.0 * 131072.0;
    let mut fm = Vec::new();
    for r in 0..=reps {
        let t = time(&p_fma, &bg_fma, wgs_fma);
        if r > 0 {
            fm.push(t);
        }
    }
    let mut fs = Vec::new();
    for r in 0..=reps {
        let t = time(&p_fma_s, &bg_fma_s, wgs_fma);
        if r > 0 {
            fs.push(t);
        }
    }
    println!("fma_scalar16 {:.0} GFLOP: median {:.3} ms = {:.1} GFLOP/s F32", flop / 1e9, median(&mut fs) * 1e3, flop / median(&mut fs) / 1e9);
    println!("fma   {:.0} GFLOP: median {:.3} ms = {:.1} GFLOP/s F32 (min {:.3} ms = {:.1} GFLOP/s)", flop / 1e9, median(&mut fm) * 1e3, flop / median(&mut fm) / 1e9, fm[0] * 1e3, flop / fm[0] / 1e9);
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

