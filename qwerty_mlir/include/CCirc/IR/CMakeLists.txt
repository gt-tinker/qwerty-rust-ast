# Yanked from the arith dialect CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS CCircAttributes.td)
mlir_tablegen(CCircOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(CCircOpsEnums.cpp.inc -gen-enum-defs)
# Currently these don't generate anything (apparently enum attrs end up in the
# enums header file), but generating them anyway to avoid grief later
mlir_tablegen(CCircOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=qcirc)
mlir_tablegen(CCircOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=qcirc)

# Yanked from mlir/IR/CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS CCircInterfaces.td)
mlir_tablegen(CCircOpsTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(CCircOpsTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(CCircOpsOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CCircOpsOpInterfaces.cpp.inc -gen-op-interface-defs)

add_mlir_dialect(CCircOps qcirc)
