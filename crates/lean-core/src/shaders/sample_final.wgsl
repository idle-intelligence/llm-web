// Temperature, top-k, softmax, top-p and the draw for one decode step, on
// one thread, from the 64 best penalised logits `topk_stage.wgsl` left in
// `cands` (value bits, id pairs, best first). Division-free and with a
// counter-based generator keyed by (seed, position), so a step's token
// depends only on its logits, the seed and its position - not on the device,
// the block size or how many steps ran before it.
//   survivors   the first `k` candidates (ties toward the lower id)
//   weights     w = exp(s - s_max), s = logit * inv_temp, summed in id order
//   top-p       candidates by weight descending (lower id first) until the
//               running sum reaches top_p * sum; the rest get weight 0
//   draw        u in [0, 1) from the generator; the first survivor, in id
//               order, whose running weight exceeds u * kept_sum
// The chosen id's history count is incremented for the next step's penalty.
struct Dims {
    k: u32,
    top_p: f32,
    inv_temp: f32,
    seed_lo: u32,
    seed_hi: u32,
    position: u32,
    vocab: u32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read> cands: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const MAXK: u32 = 64u;

var<private> ids: array<u32, 64>;
var<private> ws: array<f32, 64>;
var<private> keep: array<bool, 64>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform in [0, 1): 24 random bits, exactly representable.
fn uniform_draw() -> f32 {
    var h = pcg(dims.seed_lo ^ 0x9e3779b9u);
    h = pcg(h ^ dims.seed_hi);
    h = pcg(h ^ dims.position);
    h = pcg(h ^ 0x85ebca6bu);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

// exp(x) for x <= 0 from multiplications and additions only: 2^floor(t)
// times a degree-7 polynomial of 2^frac(t). Below -80 the weight is 0.
fn exp_det(x: f32) -> f32 {
    if (x < -80.0) {
        return 0.0;
    }
    let t = x * 1.4426950408889634;
    let n = floor(t);
    let f = t - n;
    var p = 1.525273e-5;
    p = p * f + 1.540353e-4;
    p = p * f + 1.333355e-3;
    p = p * f + 9.618129e-3;
    p = p * f + 5.550411e-2;
    p = p * f + 2.402265e-1;
    p = p * f + 6.931472e-1;
    p = p * f + 1.0;
    let e = bitcast<f32>(u32(i32(n) + 127) << 23u);
    return p * e;
}

@compute @workgroup_size(1)
fn main() {
    let k = min(dims.k, MAXK);
    // Survivors by id: insertion sort of the first k candidates.
    var raw: array<f32, 64>;
    for (var i = 0u; i < k; i = i + 1u) {
        let v = max(bitcast<f32>(cands[2u * i]), -1.0e30) * dims.inv_temp;
        let id = cands[2u * i + 1u];
        var j = i;
        loop {
            if (j == 0u || ids[j - 1u] <= id) {
                break;
            }
            ids[j] = ids[j - 1u];
            raw[j] = raw[j - 1u];
            j = j - 1u;
        }
        ids[j] = id;
        raw[j] = v;
    }
    var m = raw[0];
    for (var i = 1u; i < k; i = i + 1u) {
        m = max(m, raw[i]);
    }
    var sum = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        let w = exp_det(raw[i] - m);
        ws[i] = w;
        keep[i] = true;
        sum = sum + w;
    }
    var kept_sum = sum;
    if (dims.top_p < 1.0) {
        for (var i = 0u; i < k; i = i + 1u) {
            keep[i] = false;
        }
        let goal = dims.top_p * sum;
        var cum = 0.0;
        kept_sum = 0.0;
        for (var r = 0u; r < k; r = r + 1u) {
            // Heaviest not yet taken; the lower index (lower id) wins ties.
            var best = MAXK;
            for (var i = 0u; i < k; i = i + 1u) {
                if (!keep[i] && (best == MAXK || ws[i] > ws[best])) {
                    best = i;
                }
            }
            keep[best] = true;
            cum = cum + ws[best];
            if (cum >= goal) {
                break;
            }
        }
        for (var i = 0u; i < k; i = i + 1u) {
            if (keep[i]) {
                kept_sum = kept_sum + ws[i];
            }
        }
    }
    let goal = uniform_draw() * kept_sum;
    var cum = 0.0;
    var chosen = ids[0];
    var last_kept = ids[0];
    var found = false;
    for (var i = 0u; i < k; i = i + 1u) {
        if (keep[i]) {
            last_kept = ids[i];
            if (!found) {
                cum = cum + ws[i];
                if (goal < cum) {
                    chosen = ids[i];
                    found = true;
                }
            }
        }
    }
    if (!found) {
        chosen = last_kept;
    }
    counts[chosen] = counts[chosen] + 1u;
    out_idx[0] = chosen;
}
