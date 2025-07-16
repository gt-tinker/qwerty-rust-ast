//===- QcircPasses.h - Qcirc Patterns and Passes ------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on MemRef operations.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H
#define DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H

#include "mlir/Pass/Pass.h"
// include func::FuncOp definition for recursion to loop pass
#include "mlir/Dialect/Func/IR/FuncOps.h"
// For lowering to LLVM
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace qcirc {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "CCirc/Transforms/CCircPasses.h.inc"

} // namespace qcirc

#endif // DIALECT_INCLUDE_CCIRC_TRANSFORMS_CCIRC_TYPES_H
