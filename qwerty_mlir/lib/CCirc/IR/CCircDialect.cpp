//===- CCircDialect.cpp - CCirc dialect ---------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"

#include "CCirc/IR/CCircDialect.h"
#include "CCirc/IR/CCircAttributes.h"
#include "CCirc/IR/CCircOps.h"

#include "CCirc/IR/CCircOpsDialect.cpp.inc"
#include "CCirc/IR/CCircOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// CCirc dialect.
//===----------------------------------------------------------------------===//

namespace {
struct CCircInlinerInterface : public mlir::DialectInlinerInterface {
    using mlir::DialectInlinerInterface::DialectInlinerInterface;

    /// All CCirc dialect ops can be inlined.
    bool isLegalToInline(mlir::Operation *, mlir::Region *, bool, mlir::IRMapping &) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                         bool wouldBeCloned, mlir::IRMapping &valueMapping) const final {
        return true;
    }
};

// Aggravating trick to create a llvm::ConstantArray in MLIR. See the
// comment in CCircOps.td for LLVMConstantArrayOp.

} // namespace

namespace qcirc {

void CCircDialect::initialize() {
    registerAttributes();
    registerTypes();

    addOperations<
#define GET_OP_LIST
#include "CCirc/IR/CCircOps.cpp.inc"
    >();

    addInterfaces<CCircInlinerInterface,
                  CCircLLVMIRTranslationInterface>();
}

void registerCCircDialectTranslation(mlir::DialectRegistry &registry) {
    // This will actually register the translation interface in
    // CCircDialect::initialize() (see above)
    registry.insert<CCircDialect>();
    // ...so we don't need to do something like this, even though the `llvm'
    // dialect in MLIR does (why? the LLVM dialect does not register this
    // interface itself as we do above):
    //registry.addExtension(+[](mlir::MLIRContext *ctx, CCircDialect *dialect) {
    //    dialect->addInterfaces<CCircLLVMIRTranslationInterface>();
    //});
}

} // namespace qcirc
