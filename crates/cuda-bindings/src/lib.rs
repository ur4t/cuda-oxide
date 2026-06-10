/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-level FFI to the CUDA Driver API (`cuda.h`).
//!
//! Bindings are generated at build time by [`bindgen`](https://docs.rs/bindgen) from `wrapper.h`,
//! which includes the toolkit `cuda.h`. The build script passes `-I$CUDA_TOOLKIT_PATH/include` to
//! Clang, emits `cargo:rustc-link-search` for discovered library directories, and links
//! `libcuda` (`dylib=cuda`). Generated Rust lives under `OUT_DIR` as `bindings.rs` and is pulled in
//! via [`include!`].
//!
//! **Toolkit path:** set `CUDA_TOOLKIT_PATH` to the root of your CUDA installation (the directory
//! that contains `include/cuda.h`). If unset, the build script and [`cuda_toolkit_dir`] both use
//! `/usr/local/cuda`. Changing `CUDA_TOOLKIT_PATH` or `wrapper.h` triggers a rebuild.
//!
//! Types and functions in the generated module are `unsafe` where required by Rust; each carries
//! the usual CUDA API preconditions (valid handles, device state, stream ordering, etc.).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// The generated bindings carry CUDA's C doxygen comments verbatim. Those contain
// `[...]` spans, bare URLs, HTML-ish tables, and `\brief`-style code that rustdoc
// flags as broken intra-doc links, bare URLs, unclosed HTML tags, and unparseable
// Rust code blocks. We keep the comments (they are useful API docs) but silence
// these lints for this generated FFI crate; its doctests are excluded from the
// `--doc` gate in CI.
#![allow(rustdoc::broken_intra_doc_links)]
#![allow(rustdoc::bare_urls)]
#![allow(rustdoc::invalid_html_tags)]
#![allow(rustdoc::invalid_rust_codeblocks)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::env;

/// Root directory of the CUDA toolkit used for this build, for host code that must agree with
/// compile-time include and link paths (e.g. loading companion libraries or probing layout).
///
/// Resolution matches `build.rs`: [`std::env::var`] on `CUDA_TOOLKIT_PATH`; on `NotPresent` or
/// `NotUnicode`, returns `/usr/local/cuda`. If the variable is set, its value is used verbatim.
pub fn cuda_toolkit_dir() -> String {
    env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string())
}
