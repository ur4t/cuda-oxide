// Minimal repro: float `!=` should be UNORDERED (x != x is TRUE for NaN).
// Rust PartialEq::ne on floats is unordered. cuda-oxide lowered it to fcmp ONE
// (ordered, FALSE for NaN), so x != x folded to false -> NaN handling broken.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn is_nan(x: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(ce) = c.get_mut(idx) {
            let v = x[i];
            // x != x is the canonical NaN check; must be 1.0 for NaN, 0.0 otherwise.
            *ce = if v != v { 1.0 } else { 0.0 };
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let x = vec![f32::NAN, 1.0, 0.0, -1.0, f32::INFINITY];
    let n = x.len();
    let xd = DeviceBuffer::from_host(&stream, &x).unwrap();
    let mut cd = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
    let m = kernels::load(&ctx).expect("load");
    m.is_nan(&stream, LaunchConfig::for_num_elems(n as u32), &xd, &mut cd).expect("launch");
    let c = cd.to_host_vec(&stream).unwrap();
    println!("input : {:?}", x);
    println!("x!=x  : {:?}  (expect [1.0, 0.0, 0.0, 0.0, 0.0])", c);
    let ok = c[0] == 1.0 && c[1] == 0.0;
    println!("{}", if ok { "SUCCESS: x != x is unordered (NaN detected)" } else { "FAILURE: x != x folded to false for NaN" });
    std::process::exit(if ok { 0 } else { 1 });
}
