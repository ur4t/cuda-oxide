// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression test for `Ord::cmp` in device code.
//!
//! Run: cargo oxide run ord_cmp

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[derive(Clone, Copy, Default, Eq, PartialEq, PartialOrd)]
struct Foo<T> {
    pieces: [T; 4],
}

impl<T> Ord for Foo<T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.pieces[0].cmp(&other.pieces[0])
    }
}

#[cuda_module]
mod kernels {
    use super::*;
    use core::cmp::Ordering;

    fn cmp_code<T>(lhs: T, rhs: T) -> i32
    where
        T: Ord + Copy + Default,
    {
        let a = Foo {
            pieces: [lhs, T::default(), T::default(), T::default()],
        };
        let b = Foo {
            pieces: [rhs, T::default(), T::default(), T::default()],
        };
        match a.cmp(&b) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }

    #[kernel]
    pub fn cmp_kernel(mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let lane = idx.get();

        let code = match lane {
            0 => cmp_code(0_u32, u32::MAX),
            1 => cmp_code(u32::MAX, u32::MAX),
            2 => cmp_code(u32::MAX, 0_u32),
            3 => cmp_code(i32::MIN, i32::MAX),
            4 => cmp_code(i32::MIN, i32::MIN),
            5 => cmp_code(i32::MAX, i32::MIN),
            6 => cmp_code(0_u64, u64::MAX),
            7 => cmp_code(u64::MAX, u64::MAX),
            8 => cmp_code(u64::MAX, 0_u64),
            9 => cmp_code(i64::MIN, i64::MAX),
            10 => cmp_code(i64::MIN, i64::MIN),
            11 => cmp_code(i64::MAX, i64::MIN),
            12 => cmp_code(0_usize, usize::MAX),
            13 => cmp_code(usize::MAX, usize::MAX),
            14 => cmp_code(usize::MAX, 0_usize),
            15 => cmp_code(isize::MIN, isize::MAX),
            16 => cmp_code(isize::MIN, isize::MIN),
            17 => cmp_code(isize::MAX, isize::MIN),
            _ => return,
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = code;
        }
    }
}

fn main() {
    println!("=== Ord cmp regression test ===");

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let module = ctx
        .load_module_from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/ord_cmp.ptx"))
        .expect("failed to load PTX");
    let module = kernels::from_module(module).expect("failed to initialize typed CUDA module");
    let stream = ctx.default_stream();

    let mut out = DeviceBuffer::<i32>::zeroed(&stream, 18).expect("failed to allocate output");
    module
        .cmp_kernel(
            stream.as_ref(),
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut out,
        )
        .expect("kernel launch failed");

    let got = out.to_host_vec(&stream).expect("failed to copy output");
    let expected = [
        -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1,
    ];
    assert_eq!(got, expected);
    println!("PASS: {got:?}");
}
