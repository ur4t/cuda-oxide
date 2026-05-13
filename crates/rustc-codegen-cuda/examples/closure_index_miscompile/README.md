# closure_index_miscompile

Minimal reproduction for a `rustc-codegen-cuda` miscompile where a closure that
takes `idx: usize` and reads a captured array through that index has the
**load's `getelementptr` lowered with `idx` replaced by literal 0**. Every call
to the closure therefore returns whatever the slot-0 value happens to be,
regardless of the argument actually passed.

The example is reduced from a CUDA kernel that indexed a two-slot wrapper
through a closure. The closure returned the slot-0 value for both `k` arms,
which later optimisation could then fold into a single path.

## What the example does

`src/main.rs` contains **14 kernel variants** that bisect the trigger
conditions:

| # | Variant | Purpose |
|---|---------|---------|
|  1 | `test_unrolled_baseline` | No closure; sanity check |
|  2 | `test_closure_indexes_into_array` | Closure indexes a raw `[f32; 2]` |
|  3 | `test_closure_indexes_via_match` | Closure uses `match`, not `[idx]` |
|  4 | `test_closure_into_struct_wrapper` | Closure indexes through `Pair::get` |
|  5 | `test_closure_pre_loaded_outside` | Closure selects pre-loaded values |
|  6 | `test_closure_node_ref_access` | Closure indexes through `Pair::node` |
|  7 | `test_closure_with_shuffle` | One warp shuffle inside the closure |
|  8 | `test_closure_two_shuffles` | Two warp shuffles inside the closure |
|  9 | `test_two_shuffles_no_indexed_load` | Two shuffles without an indexed load |
| 10 | `test_two_shuffles_raw_array` | Two shuffles with a raw array |
| 11 | `test_two_shuffles_no_transparent` | Two shuffles with a non-transparent wrapper |
| 12 | `test_two_shuffles_inlined` | Same indexed access without a closure binding |
| 13 | `test_closure_shuffle_with_captures` | Two shuffles plus extra captures |
| 14 | `test_closure_via_array_literal` | Results collected through `[compute(0), compute(1)]` |

Each kernel writes `compute(1) - compute(0)` for inputs chosen so a correct
implementation must produce a non-zero difference (`+5.0` for every element in
this example). The kernel prints
`PASS` if the diff is non-zero, `FAIL (diff = 0)` if the miscompile triggers.

## Trigger conditions

After bisection, the miscompile fires when **all of** the following are true:

1. The index-by-`usize` happens inside a Rust closure (not the surrounding
   function body).
2. The closure has ≥ 1 captured upvar (so it lowers as a function over
   `&Self`).
3. The closure body uses warp shuffles in a way that prevents LLVM from
   inlining it back into the caller (in practice: ≥ 2 calls, where
   `llvm.nvvm.shfl.sync.*` is marked `convergent`).
4. The array being indexed is reached **through a struct field projection**
   (e.g. `Pair(pub [f32; 2])`). Bare `[f32; 2]` upvars are lowered correctly.

When all four hold, the rustc MIR

```text
_4 = &((*_9).0: [f32; 2])[_2]
```

(place projection chain `[Deref, Field(0, [f32;2]), Index(_2)]`) is silently
truncated by the mir-importer to a `[Deref, Field(0)]` address, dropping the
runtime `Index`.

## Root cause

In `crates/mir-importer/src/translator/rvalue.rs`, the `Rvalue::Ref` arm has
five cases. Case 2 (`[Deref, Field, …]`) emits a `MirFieldAddrOp` for the
first field, then walks the remaining projections in an inner loop that only
handles further `Field`s — every other variant hits `_ => break`. After the
loop, the function unconditionally returns the partial field address,
**silently discarding** any tail projections, including a runtime `Index`.

The fix delegates the tail walk to the existing
`translate_place_addr_from_slot` helper (which is now also extended to handle
runtime `Index` by emitting `MirArrayElementAddrOp`). With the fix, the same
MIR lowers to

```text
%v7 = getelementptr inbounds { [2 x float] }, ptr %v5, i32 0, i32 0   ; field 0
%v8 = getelementptr inbounds [2 x float], ptr %v7, i32 0, i64 %v3     ; index
%v9 = load float, ptr %v8                                              ; correct
```

and all 14 kernels report `PASS`.

## Build & run

From the cuda-oxide repository root:

```bash
cargo oxide run closure_index_miscompile
cargo oxide pipeline closure_index_miscompile    # dump MIR + LLVM IR
```

Requires a CUDA-capable GPU and the cuda-oxide rustc toolchain.
