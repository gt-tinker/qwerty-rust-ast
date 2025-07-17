# Yanked from the arith dialect CMakeLists.txt
set(LLVM_TARGET_DEFINITIONS CCircAttributes.td)
# Currently these don't generate anything (apparently enum attrs end up in the
# enums header file), but generating them anyway to avoid grief later
mlir_tablegen(CCircOpsAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=qcirc)
mlir_tablegen(CCircOpsAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=qcirc)

add_mlir_dialect(CCircOps qcirc)
