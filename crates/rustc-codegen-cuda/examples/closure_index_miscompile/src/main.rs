// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Minimal reproduction for a rustc-codegen-cuda closure-indexing miscompile
 * found while reducing a CUDA kernel that indexed a two-slot wrapper through
 * a closure.
 *
 * Symptom: in
 *
 *     let compute = |idx: usize| { ... arr[idx] ... };
 *     [compute(0), compute(1)]
 *
 * both invocations produce identical results, equal to what `compute(0)`
 * should produce. The `idx` parameter is silently ignored at the load
 * site.
 *
 * The corresponding LLVM IR (--emit llvm-ir) shows:
 *
 *     %v3 = phi i64 [ %v1, %entry ]                       ; idx
 *     %v6 = icmp ult i64 %v3, 2                           ; bounds check OK
 *     br i1 %v6, label %bb1, label %bb4
 *   bb1:
 *     %v7 = getelementptr inbounds { [2 x float] }, ptr %v5, i32 0, i32 0
 *                                                          ; ^ idx ignored
 *     %v9 = load float, ptr %v8                           ; always slot 0
 *
 * Each kernel below writes the *difference* compute(1) - compute(0). If
 * the closure indexing works the difference is non-zero (input designed
 * so it must differ); if the miscompile triggers the difference is 0.
 *
 * Build:
 *     cargo oxide run closure_index_miscompile
 *
 * Before the fix, the variants that route a runtime index through the tail
 * of a field-projected `Rvalue::Ref` report diff = 0. With the fix, all
 * variants report diff = 5.
 */

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// 2-element struct wrapper. `#[repr(transparent)]` keeps the inner array
/// layout preserved while still exercising a field projection before the
/// indexed access.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Pair(pub [f32; 2]);

impl Pair {
    /// Returns the inner element by value. Matches the `Pair`-style API
    /// used in the original "simple" repro attempts.
    #[inline(always)]
    pub fn get(&self, i: usize) -> f32 {
        self.0[i]
    }

    /// Returns a reference to the inner element. This exercises the
    /// `wrapper.node(slot) -> &T` API shape that exposed the importer bug.
    #[inline(always)]
    pub const fn node(&self, slot: usize) -> &f32 {
        &self.0[slot]
    }
}

/// Same as `Pair`, but WITHOUT `#[repr(transparent)]`. Used to check
/// whether the miscompile is triggered specifically by `repr(transparent)`
/// or by any newtype wrapper over `[f32; 2]`.
#[derive(Copy, Clone)]
pub struct PairNoTransparent(pub [f32; 2]);

impl PairNoTransparent {
    #[inline(always)]
    pub const fn node(&self, slot: usize) -> &f32 {
        &self.0[slot]
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Baseline: explicit slot-0 / slot-1 unroll, no closure. Always works.
    #[kernel]
    pub fn test_unrolled_baseline(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = input[i];

        let r0 = pair[0] + 1.0;
        let r1 = pair[1] + 1.0;

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// THE BUG: closure indexes `arr[idx]` with `idx` being the closure
    /// parameter. Called twice with literal 0 and 1.
    #[kernel]
    pub fn test_closure_indexes_into_array(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair: [f32; 2] = input[i];

        let compute = |k: usize| -> f32 { pair[k] + 1.0 };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Variation: closure does an `if k == 0` match instead of indexing.
    /// Tests whether the bug is specifically about the indexed GEP path or
    /// about closure-with-param in general. Expected: PASS — the match
    /// path doesn't go through the GEP-pinned-to-0 codegen.
    #[kernel]
    pub fn test_closure_indexes_via_match(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair: [f32; 2] = input[i];

        let compute = |k: usize| -> f32 {
            let v = if k == 0 { pair[0] } else { pair[1] };
            v + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Indexed access through a `#[repr(transparent)]` struct wrapping
    /// `[f32; 2]`, with the indexing hidden behind a method call.
    #[kernel]
    pub fn test_closure_into_struct_wrapper(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 { pair.get(k) + 1.0 };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Indexed access goes through `pair.node(k)` returning `&f32`, then
    /// dereferences the result. Same as `test_closure_into_struct_wrapper`
    /// but the accessor returns a reference instead of by-value.
    #[kernel]
    pub fn test_closure_node_ref_access(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 { *pair.node(k) + 1.0 };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Closer to the actual failing case: closure body contains a warp
    /// shuffle, mimicking `tmp0 = warp::shuffle_xor_f32(v, ...)`. The
    /// hypothesis is that the buggy GEP-pin-to-zero only fires when the
    /// closure body contains certain other ops (shuffle intrinsic call,
    /// etc.) that block inlining/SROA from cleaning up the bad GEP.
    #[kernel]
    pub fn test_closure_with_shuffle(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let shuf = warp::shuffle_xor_f32(v, 1);
            v + shuf * 0.0 + 1.0 // shuf contribution = 0 so per-thread answer is still v + 1
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: 2 shuffles, NO captures. Same straight-line topology
    /// as `test_closure_with_shuffle` but with a *second* shuffle call.
    #[kernel]
    pub fn test_closure_two_shuffles(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1); // = v (XOR with same lane back)
            // Per-thread answer ends up being v + 0 + 1 = v + 1.
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: 2 shuffles but operand is `1.0` constant — NOT loaded
    /// from `pair.node(k)`. Does the closure parameter even get used?
    #[kernel]
    pub fn test_two_shuffles_no_indexed_load(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        // The closure parameter `k` is used ONLY in this `if` to pick a
        // pre-loaded value. No GEP indexed by `k`.
        let a = pair.0[0];
        let b = pair.0[1];

        let compute = |k: usize| -> f32 {
            let v = if k == 0 { a } else { b };
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1);
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: 2 shuffles, indexed load goes through a *bare array*
    /// `[f32; 2]` rather than the `Pair` newtype. Tests whether the bug
    /// is wrapper-specific or also reproduces on raw arrays.
    #[kernel]
    pub fn test_two_shuffles_raw_array(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let arr: [f32; 2] = input[i];

        let compute = |k: usize| -> f32 {
            let v = arr[k];
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1);
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    // (test_two_sqrt_calls removed — it caused PTX gen to silently fail.)

    /// Bisection: 2 shuffles, indexed load through a non-`repr(transparent)`
    /// newtype. Tests whether the bug is specifically about `repr(transparent)`.
    #[kernel]
    pub fn test_two_shuffles_no_transparent(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = PairNoTransparent(input[i]);

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let tmp1 = warp::shuffle_xor_f32(tmp0, 1);
            v + tmp1 * 0.0 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Bisection: like `test_closure_two_shuffles` but the closure is
    /// INLINED at the call site (no closure binding). Tests whether the
    /// bug is about the closure body or about the indexed access in any
    /// caller-with-param pattern.
    #[kernel]
    pub fn test_two_shuffles_inlined(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        // No closure: two straight-line bodies, but each does indexed access
        // through pair.node(k) where k is a *runtime* value coming from a
        // local var (to defeat constant propagation at the access site).
        let mut k = 0usize;
        let v0 = *pair.node(k);
        let tmp0_0 = warp::shuffle_xor_f32(v0, 1);
        let tmp1_0 = warp::shuffle_xor_f32(tmp0_0, 1);
        let r0 = v0 + tmp1_0 * 0.0 + 1.0;

        k = 1;
        let v1 = *pair.node(k);
        let tmp0_1 = warp::shuffle_xor_f32(v1, 1);
        let tmp1_1 = warp::shuffle_xor_f32(tmp0_1, 1);
        let r1 = v1 + tmp1_1 * 0.0 + 1.0;

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Stronger trigger: closure body has a shuffle plus several constants
    /// captured from the enclosing scope. This keeps the same straight-line
    /// topology as the reduced failure while staying domain-neutral.
    #[kernel]
    pub fn test_closure_shuffle_with_captures(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        // Captures with non-trivial use of the indexed load.
        let w_0: f32 = 0.0;
        let w_1: f32 = 1.0;
        let w2_0: f32 = 0.0;
        let w2_1: f32 = 0.0;

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let out0 = v * w_1 + tmp0 * w_0;
            let to_shfl = v * w2_1 + tmp0 * w2_0;
            let tmp1 = warp::shuffle_xor_f32(to_shfl, 1);
            out0 + tmp1 + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }

    /// Like `test_closure_shuffle_with_captures` but with the call-site
    /// idiom `[compute(0), compute(1)]` (constructing an array literal).
    #[kernel]
    pub fn test_closure_via_array_literal(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair = Pair(input[i]);

        let w_0: f32 = 0.0;
        let w_1: f32 = 1.0;
        let w2_0: f32 = 0.0;
        let w2_1: f32 = 0.0;

        let compute = |k: usize| -> f32 {
            let v = *pair.node(k);
            let tmp0 = warp::shuffle_xor_f32(v, 1);
            let out0 = v * w_1 + tmp0 * w_0;
            let to_shfl = v * w2_1 + tmp0 * w2_0;
            let tmp1 = warp::shuffle_xor_f32(to_shfl, 1);
            out0 + tmp1 + 1.0
        };

        let pair_out: [f32; 2] = [compute(0), compute(1)];

        if let Some(slot) = out.get_mut(idx) {
            *slot = pair_out[1] - pair_out[0];
        }
    }

    /// Control: closure captures the pre-loaded values, not the array.
    /// No GEP indexed by closure param. Expected: PASS.
    #[kernel]
    pub fn test_closure_pre_loaded_outside(input: &[[f32; 2]], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= input.len() {
            return;
        }
        let pair: [f32; 2] = input[i];
        let a = pair[0];
        let b = pair[1];

        let compute = |k: usize| -> f32 {
            let v = if k == 0 { a } else { b };
            v + 1.0
        };

        let r0 = compute(0);
        let r1 = compute(1);

        if let Some(slot) = out.get_mut(idx) {
            *slot = r1 - r0;
        }
    }
}

const N: usize = 4;

fn make_input() -> Vec<[f32; 2]> {
    // Design: arr[1] - arr[0] = +5.0 for every thread, so any correct
    // kernel reports diff = 5.0. The miscompile reports diff = 0.0.
    (0..N).map(|i| [i as f32, i as f32 + 5.0]).collect()
}

fn run_and_report<F>(name: &str, stream: &Arc<CudaStream>, launch: F)
where
    F: FnOnce(&Arc<CudaStream>, LaunchConfig, &DeviceBuffer<[f32; 2]>, &mut DeviceBuffer<f32>),
{
    let input = make_input();
    let dev_in = DeviceBuffer::from_host(stream, &input).unwrap();
    let mut dev_out = DeviceBuffer::<f32>::zeroed(stream, N).unwrap();

    launch(
        stream,
        LaunchConfig::for_num_elems(N as u32),
        &dev_in,
        &mut dev_out,
    );

    let host_out = dev_out.to_host_vec(stream).unwrap();
    let all_correct = host_out.iter().all(|&d| (d - 5.0).abs() < 1e-6);
    let any_zero = host_out.iter().any(|&d| d.abs() < 1e-6);

    let verdict = if all_correct {
        "PASS"
    } else if any_zero {
        "FAIL (miscompile: closure idx ignored)"
    } else {
        "FAIL (unexpected)"
    };
    println!(
        "  {name:<40}  {verdict}    (diff[0..{}] = {:?})",
        N.min(4),
        &host_out[..N.min(4)]
    );
}

fn main() {
    println!("=== Closure-index miscompile minimal repro ===\n");

    let ctx = CudaContext::new(0).expect("CUDA init");
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Load embedded PTX");

    println!("Each kernel computes r1 - r0 where r0 reads arr[0] and r1 reads arr[1].");
    println!("Input designed so the correct diff is +5.0 for every element.\n");

    run_and_report("test_unrolled_baseline", &stream, |s, cfg, i, o| {
        module
            .test_unrolled_baseline(s, cfg, i, o)
            .expect("launch")
    });
    run_and_report(
        "test_closure_indexes_into_array",
        &stream,
        |s, cfg, i, o| {
            module
                .test_closure_indexes_into_array(s, cfg, i, o)
                .expect("launch")
        },
    );
    run_and_report(
        "test_closure_indexes_via_match",
        &stream,
        |s, cfg, i, o| {
            module
                .test_closure_indexes_via_match(s, cfg, i, o)
                .expect("launch")
        },
    );
    run_and_report(
        "test_closure_into_struct_wrapper",
        &stream,
        |s, cfg, i, o| {
            module
                .test_closure_into_struct_wrapper(s, cfg, i, o)
                .expect("launch")
        },
    );
    run_and_report(
        "test_closure_pre_loaded_outside",
        &stream,
        |s, cfg, i, o| {
            module
                .test_closure_pre_loaded_outside(s, cfg, i, o)
                .expect("launch")
        },
    );
    run_and_report("test_closure_node_ref_access", &stream, |s, cfg, i, o| {
        module
            .test_closure_node_ref_access(s, cfg, i, o)
            .expect("launch")
    });
    run_and_report("test_closure_with_shuffle", &stream, |s, cfg, i, o| {
        module
            .test_closure_with_shuffle(s, cfg, i, o)
            .expect("launch")
    });
    run_and_report("test_closure_two_shuffles", &stream, |s, cfg, i, o| {
        module
            .test_closure_two_shuffles(s, cfg, i, o)
            .expect("launch")
    });
    run_and_report(
        "test_two_shuffles_no_indexed_load",
        &stream,
        |s, cfg, i, o| {
            module
                .test_two_shuffles_no_indexed_load(s, cfg, i, o)
                .expect("launch")
        },
    );
    run_and_report("test_two_shuffles_raw_array", &stream, |s, cfg, i, o| {
        module
            .test_two_shuffles_raw_array(s, cfg, i, o)
            .expect("launch")
    });
    run_and_report(
        "test_two_shuffles_no_transparent",
        &stream,
        |s, cfg, i, o| {
            module
                .test_two_shuffles_no_transparent(s, cfg, i, o)
                .expect("launch")
        },
    );
    run_and_report("test_two_shuffles_inlined", &stream, |s, cfg, i, o| {
        module
            .test_two_shuffles_inlined(s, cfg, i, o)
            .expect("launch")
    });
    run_and_report(
        "test_closure_shuffle_with_captures",
        &stream,
        |s, cfg, i, o| {
            module
                .test_closure_shuffle_with_captures(s, cfg, i, o)
                .expect("launch")
        },
    );
    run_and_report(
        "test_closure_via_array_literal",
        &stream,
        |s, cfg, i, o| {
            module
                .test_closure_via_array_literal(s, cfg, i, o)
                .expect("launch")
        },
    );
}
