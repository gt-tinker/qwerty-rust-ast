//add_mlir_dialect_library(MLIRCCircTransforms
  DISABLE_INSTALL


  DEPENDS
  MLIRCCircPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRCCirc
  MLIRPass
  MLIRTransforms
  MLIRNVVMDialect #we reference NVVM barrier ops
  MLIRAffineToStandard
  MLIRFuncInlinerExtension 
  MLIRSCFToControlFlow
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
)

