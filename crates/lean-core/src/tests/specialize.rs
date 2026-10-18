//! Build constants are written into the shader source (`engine::specialize`),
//! never left to WGSL `override` evaluation at pipeline creation.

use crate::engine::specialize;

#[test]
fn specialize_rewrites_only_the_named_consts() {
    let src = "const HEAD_DIM: u32 = 64u;\nconst DV: u32 = HEAD_DIM / 4u;\nconst CACHE: bool = false;\n";
    let out = specialize(src, &[("HEAD_DIM", 128.0), ("CACHE", 1.0)]);
    assert_eq!(out, "const HEAD_DIM: u32 = 128u;\nconst DV: u32 = HEAD_DIM / 4u;\nconst CACHE: bool = true;\n");
    assert_eq!(specialize(src, &[]), src);
}

#[test]
fn no_shader_uses_override() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let src = std::fs::read_to_string(&path).unwrap();
        assert!(!src.lines().any(|l| l.trim_start().starts_with("override ")), "{} declares an override", path.display());
    }
}
