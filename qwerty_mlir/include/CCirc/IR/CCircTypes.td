//===- CCircOps.td - CCirc dialect ops -----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD
#define DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD

include "CCirc/IR/CCircDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// class OperationType

//===----------------------------------------------------------------------===//
// Base qcirc type definition.
//===----------------------------------------------------------------------===//

class CCirc_Type<string name, string typeMnemonic, list<Trait> traits = []> :
        TypeDef<CCirc_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def CCirc_Wire : CCirc_Type<"Wire", "wire"> {
    let summary = "Wire";
    let description = [{
        Opaque type for a single two-level quantum state
    }];
}

def CCirc_Array : CCirc_Type<"Array", "array"> {
    let summary = "Array of values";
    let description = [{
        Array of elemType. Lowered to a 1D QIR array
    }];
    let parameters = (ins "::mlir::Type":$elemType, "uint64_t":$dim);
    let assemblyFormat = "`<` $elemType `>` `` `[` $dim `]`";
}

def CCirc_Callable : CCirc_Type<"Callable", "callable"> {
    let summary = "Callable value";
    let description = [{
        Pointer to a buffer containing captures and a function pointer. Lowered
        to a a QIR callable
    }];
    let parameters = (ins "::mlir::FunctionType":$functionType);
    let assemblyFormat = "$functionType";
}

#endif // DIALECT_INCLUDE_CCIRC_IR_CCIRC_TYPES_TD
