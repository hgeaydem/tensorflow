/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/spmd/shardy/round_trip_common/export_backend_func_calls.h"

#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayAttr;
using ::mlir::DictionaryAttr;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::OpConversionPattern;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::NamedComputationOp;
using ::mlir::sdy::TensorShardingPerValueAttr;

class BackendNamedComputationPattern
    : public OpConversionPattern<NamedComputationOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      NamedComputationOp namedComputationOp, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    DictionaryAttr frontendAttrs = getFrontendAttrs(namedComputationOp);
    if (!frontendAttrs || !frontendAttrs.contains(kXlaBackendConfigAttr)) {
      return mlir::failure();
    };

    auto moduleOp = namedComputationOp->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToEnd(&moduleOp.getRegion().front());
    auto funcOp = rewriter.create<FuncOp>(
        namedComputationOp.getLoc(), namedComputationOp.getName(),
        rewriter.getFunctionType(
            namedComputationOp.getBody().getArgumentTypes(),
            namedComputationOp.getResultTypes()),
        rewriter.getStringAttr("private"),
        /*argAttrs=*/ArrayAttr(), /*resultAttrs=*/ArrayAttr());
    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);
    mlir::StringAttr funcName = symbolTable.insert(funcOp);
    rewriter.setInsertionPointToStart(funcOp->getBlock());
    mlir::sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
        namedComputationOp.getBody(), funcOp.getBody(), rewriter);
    rewriter.setInsertionPoint(namedComputationOp);

    // Copy the input shardings to the func.
    if (std::optional<TensorShardingPerValueAttr> inShardings =
            namedComputationOp.getInShardings()) {
      for (auto [i, sharding] : llvm::enumerate(inShardings->getShardings())) {
        funcOp.setArgAttr(i, kShardingAttr, sharding);
      }
    }

    // Copy the output shardings to the func AND call.
    mlir::SmallVector<NamedAttribute> callOpAttrs(
        namedComputationOp->getDiscardableAttrs());
    if (std::optional<TensorShardingPerValueAttr> outShardings =
            namedComputationOp.getOutShardings()) {
      for (auto [i, sharding] : llvm::enumerate(outShardings->getShardings())) {
        funcOp.setResultAttr(i, kShardingAttr, sharding);
      }
      callOpAttrs.push_back(
          NamedAttribute(rewriter.getStringAttr(kShardingAttr), *outShardings));
    }
    // Add the out_shardings to the call op.

    auto callOp = rewriter.replaceOpWithNewOp<CallOp>(
        namedComputationOp, namedComputationOp.getResultTypes(), funcName,
        adaptor.getOperands());
    callOp->setAttrs(callOpAttrs);

    return mlir::success();
  }
};

// Converts a `NamedComputationOp` with `backend_config` into a `CallOp`.
class ExportBackendFuncCallsPass
    : public mlir::PassWrapper<ExportBackendFuncCallsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportBackendFuncCallsPass)

  void runOnOperation() final {
    mlir::MLIRContext& context = getContext();
    mlir::ConversionTarget target(context);
    target.addLegalOp<FuncOp, mlir::func::ReturnOp, CallOp>();
    target.addDynamicallyLegalOp<NamedComputationOp>([](NamedComputationOp op) {
      DictionaryAttr frontendAttrs = getFrontendAttrs(op);
      return !(frontendAttrs && frontendAttrs.contains(kXlaBackendConfigAttr));
    });
    mlir::RewritePatternSet patterns(&context);
    patterns.add<BackendNamedComputationPattern>(&context);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-export-backend-func-calls";
  }

  StringRef getDescription() const override {
    return "Creates a pass that converts a `NamedComputationOp` with a "
           "`backend_config` attr to a `CallOp` with a new private function "
           "called the `NamedComputationOp`'s `name`. The new `FuncOp` and "
           "`CallOp` have the same shardings as the original "
           "`NamedComputationOp`s operands/results.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createExportBackendFuncCallsPass() {
  return std::make_unique<ExportBackendFuncCallsPass>();
}

void registerExportBackendFuncCallsPass() {
  mlir::registerPass(createExportBackendFuncCallsPass);
}

}  // namespace sdy
}  // namespace xla
