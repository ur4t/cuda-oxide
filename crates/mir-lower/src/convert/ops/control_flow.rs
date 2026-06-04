/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Control flow operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts `dialect-mir` terminators and control flow operations to their
//! LLVM dialect equivalents.
//!
//! # Operations
//!
//! | MIR Operation      | LLVM Operation              | Description             |
//! |--------------------|-----------------------------|-------------------------|
//! | `mir.return`       | `llvm.return`               | Function return         |
//! | `mir.goto`         | `llvm.br`                   | Unconditional branch    |
//! | `mir.cond_branch`  | `llvm.cond_br`              | Conditional branch      |
//! | `mir.assert`       | `llvm.cond_br` + abort blk  | Runtime assertion       |
//! | `mir.unreachable`  | `llvm.unreachable`          | Unreachable marker      |
//!
//! # Block Handling
//!
//! With `DialectConversion` + `inline_region`, blocks are the ORIGINALS (moved,
//! not copied). Successor pointers are already valid — no block map lookup needed.

use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

/// Convert `mir.return` to `llvm.return`.
///
/// Handles:
/// - Void returns (no operands)
/// - Single value returns
/// - Empty struct returns (treated as void)
pub(crate) fn convert_return(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let ret_val = match operands.as_slice() {
        [] => None,
        [val] => {
            let ty = val.get_type(ctx);

            let is_empty_struct = ty
                .deref(ctx)
                .downcast_ref::<llvm_export::types::StructType>()
                .is_some_and(|st| st.num_fields() == 0);

            if ty.deref(ctx).is::<llvm_export::types::VoidType>() || is_empty_struct {
                None
            } else {
                Some(*val)
            }
        }
        _ => {
            return pliron::input_err_noloc!("Return with multiple operands not supported");
        }
    };

    let llvm_ret = llvm::ReturnOp::new(ctx, ret_val);
    rewriter.insert_operation(ctx, llvm_ret.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.unreachable` to `llvm.unreachable`.
pub(crate) fn convert_unreachable(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let unreachable_op = llvm::UnreachableOp::new(ctx);
    rewriter.insert_operation(ctx, unreachable_op.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.cond_branch` to `llvm.cond_br`.
///
/// MIR conditional branches have:
/// - Operand 0: condition (i1)
/// - Operands 1..N: arguments for true block
/// - Operands N+1..M: arguments for false block
/// - Successor 0: true block
/// - Successor 1: false block
///
/// With `inline_region`, successors are the original blocks — no block map needed.
pub(crate) fn convert_cond_branch(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let cond = match operands.first() {
        Some(v) => *v,
        None => return pliron::input_err_noloc!("CondBranch requires at least 1 operand"),
    };

    let successors: Vec<_> = op.deref(ctx).successors().collect();
    let (true_block, false_block) = match successors.as_slice() {
        [t, f] => (*t, *f),
        _ => return pliron::input_err_noloc!("CondBranch requires exactly 2 successors"),
    };

    let num_true_args = true_block.deref(ctx).arguments().count();
    let num_false_args = false_block.deref(ctx).arguments().count();

    if operands.len() != 1 + num_true_args + num_false_args {
        return pliron::input_err_noloc!(
            "CondBranch operand count mismatch. Expected {}, got {}",
            1 + num_true_args + num_false_args,
            operands.len()
        );
    }

    let true_args = operands[1..1 + num_true_args].to_vec();
    let false_args = operands[1 + num_true_args..].to_vec();

    let llvm_br = llvm::CondBrOp::new(ctx, cond, true_block, true_args, false_block, false_args);
    rewriter.insert_operation(ctx, llvm_br.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert `mir.assert` to conditional branch with abort block.
///
/// MIR assert is converted to:
/// 1. Create an abort block with `llvm.unreachable`
/// 2. `llvm.cond_br` to success block (if true) or abort block (if false)
///
/// The abort block is inserted directly (not through the rewriter), since it's
/// a new block, not a replacement for anything.
pub(crate) fn convert_assert(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (cond, args) = match operands.as_slice() {
        [cond, args @ ..] => (*cond, args),
        _ => return pliron::input_err_noloc!("Assert requires at least 1 operand"),
    };

    let successors: Vec<_> = op.deref(ctx).successors().collect();
    let success_block = match successors.as_slice() {
        [blk] => *blk,
        _ => return pliron::input_err_noloc!("Assert requires exactly 1 successor"),
    };

    let region = match op
        .deref(ctx)
        .get_parent_block()
        .and_then(|b| b.deref(ctx).get_parent_region())
    {
        Some(r) => r,
        None => return pliron::input_err_noloc!("Block has no parent region"),
    };

    let abort_block = BasicBlock::new(ctx, None, vec![]);
    abort_block.insert_at_back(region, ctx);

    llvm::UnreachableOp::new(ctx)
        .get_operation()
        .insert_at_back(abort_block, ctx);

    let llvm_br = llvm::CondBrOp::new(ctx, cond, success_block, args.to_vec(), abort_block, vec![]);
    rewriter.insert_operation(ctx, llvm_br.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert `mir.goto` to `llvm.br`.
///
/// Handles ZST (Zero-Sized Type) padding: if the destination block expects
/// more arguments than provided, missing arguments for empty struct types
/// are filled with `undef` values.
///
/// With `inline_region`, the dest block arg types are already converted by
/// the framework, so the ZST check works on LLVM types directly.
pub(crate) fn convert_goto(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let successors: Vec<_> = op.deref(ctx).successors().collect();
    let dest = match successors.as_slice() {
        [dest] => *dest,
        _ => return pliron::input_err_noloc!("Goto requires exactly 1 successor"),
    };

    let mut final_args: Vec<_> = op.deref(ctx).operands().collect();

    let num_dest_args = dest.deref(ctx).arguments().count();

    if final_args.len() < num_dest_args {
        let dest_args: Vec<_> = dest.deref(ctx).arguments().skip(final_args.len()).collect();

        for dest_arg in dest_args {
            let arg_ty = dest_arg.get_type(ctx);

            let is_empty_struct = arg_ty
                .deref(ctx)
                .downcast_ref::<llvm_export::types::StructType>()
                .is_some_and(|st| st.num_fields() == 0);

            if is_empty_struct {
                let undef = llvm::UndefOp::new(ctx, arg_ty);
                rewriter.insert_operation(ctx, undef.get_operation());
                final_args.push(undef.get_operation().deref(ctx).get_result(0));
            } else {
                return pliron::input_err_noloc!(
                    "Goto operand count mismatch. Expected {}, got {}. \
                     Missing argument is not a ZST.",
                    num_dest_args,
                    final_args.len()
                );
            }
        }
    } else if final_args.len() > num_dest_args {
        return pliron::input_err_noloc!(
            "Goto operand count mismatch. Expected {}, got {}",
            num_dest_args,
            final_args.len()
        );
    }

    let llvm_br = llvm::BrOp::new(ctx, dest, final_args);
    rewriter.insert_operation(ctx, llvm_br.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for control flow conversion
}
