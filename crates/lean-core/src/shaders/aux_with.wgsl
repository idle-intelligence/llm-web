@group(0) @binding(6) var<storage, read> aux: array<u32>;
fn aux_at(i: u32) -> u32 {
    return aux[i];
}
