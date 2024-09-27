/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/bridge_logger.h"

#include <memory>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Define test modules that are deserialized to module ops.
static const char *const module_with_add =
    R"(module {
func.func @main(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  func.return %0 : tensor<3x4x5xf32>
}
}
)";

static const char *const module_with_sub =
    R"(module {
func.func @main(%arg0: tensor<7x8x9xi8>, %arg1: tensor<7x8x9xi8>) -> tensor<7x8x9xi8> {
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<7x8x9xi8>, tensor<7x8x9xi8>) -> tensor<7x8x9xi8>
  func.return %0 : tensor<7x8x9xi8>
}
}
)";

void UnsetEnvironmentVariables() {
  unsetenv("MLIR_BRIDGE_LOG_PASS_FILTER");
  unsetenv("MLIR_BRIDGE_LOG_STRING_FILTER");
  unsetenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");
}

class BridgeLoggerFilters : public ::testing::Test {
 protected:
  void SetUp() override { UnsetEnvironmentVariables(); }
};

// Test pass filter.
TEST_F(BridgeLoggerFilters, TestPassFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> partitioning_pass =
      mlir::TFTPU::CreateTPUResourceReadsWritesPartitioningPass();
  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();
  std::unique_ptr<mlir::Pass> inliner_pass = mlir::createInlinerPass();

  // partitioning_pass and shape_inference_pass should match the filter,
  // inliner_pass should not.
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER",
         "TPUResourceReadsWritesPartitioningPass;TensorFlowShapeInferencePass",
         1);
  BridgeLoggerConfig logger_config;
  EXPECT_TRUE(logger_config.ShouldPrint(partitioning_pass.get(),
                                        mlir_module_with_add.get()));
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));
  EXPECT_FALSE(logger_config.ShouldPrint(inliner_pass.get(),
                                         mlir_module_with_add.get()));
}

// Test string filter.
TEST_F(BridgeLoggerFilters, TestStringFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add, mlir_module_with_sub;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));
  TF_ASSERT_OK(DeserializeMlirModule(module_with_sub, &mlir_context,
                                     &mlir_module_with_sub));
  // The pass is not relevant for this test since we don't define a pass filter.
  std::unique_ptr<mlir::Pass> dummy_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // One string appears in both modules and the other one not.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "func @main(%arg0: tensor;XXX", 1);
  BridgeLoggerConfig logger_config1;
  EXPECT_TRUE(
      logger_config1.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_TRUE(
      logger_config1.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));

  // Both strings do not appear in any module.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "func @main(%arg0:tensor;XXX", 1);
  BridgeLoggerConfig logger_config2;
  EXPECT_FALSE(
      logger_config2.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_FALSE(
      logger_config2.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));

  // String appears in one module but not in the other.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER",
         "\"tf.AddV2\"(%arg0, %arg1) : (tensor<3x4x5xf32>", 1);
  BridgeLoggerConfig logger_config3;
  EXPECT_TRUE(
      logger_config3.ShouldPrint(dummy_pass.get(), mlir_module_with_add.get()));
  EXPECT_FALSE(
      logger_config3.ShouldPrint(dummy_pass.get(), mlir_module_with_sub.get()));
}

// Test enable only top level passes filter.
TEST_F(BridgeLoggerFilters, TestEnableOnlyTopLevelPassesFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);

  // Deserialize the module with an Add operation.
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // Set the environment variable to enable only top-level passes.
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "1", 1);

  BridgeLoggerConfig logger_config;

  // Test that ShouldPrint returns true for the top-level module operation.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));

  // Find the nested function operation within the module.
  mlir::Operation *func_op = nullptr;
  for (auto &op : mlir_module_with_add->getOps()) {
    if (llvm::isa<mlir::func::FuncOp>(&op)) {
      func_op = &op;
      break;
    }
  }
  ASSERT_NE(func_op, nullptr);

  // Test that ShouldPrint returns false for the nested function operation.
  EXPECT_FALSE(logger_config.ShouldPrint(shape_inference_pass.get(), func_op));

  // Unset the environment variable to disable suppressing nested passes.
  unsetenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");

  BridgeLoggerConfig logger_config_no_suppress;

  // Test that ShouldPrint now returns true for the nested function operation.
  EXPECT_TRUE(logger_config_no_suppress.ShouldPrint(shape_inference_pass.get(),
                                                    func_op));
}

// Additional tests for various possible values of
// MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES.
TEST_F(BridgeLoggerFilters, TestEnableOnlyTopLevelPassesEnvVarValues) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // Test with environment variable set to "TRUE".
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "TRUE", 1);
  BridgeLoggerConfig logger_config_true;
  EXPECT_TRUE(logger_config_true.ShouldPrint(shape_inference_pass.get(),
                                             mlir_module_with_add.get()));

  // Test that nested operations are not printed.
  mlir::Operation *func_op = nullptr;
  for (auto &op : mlir_module_with_add->getOps()) {
    if (llvm::isa<mlir::func::FuncOp>(&op)) {
      func_op = &op;
      break;
    }
  }
  ASSERT_NE(func_op, nullptr);
  EXPECT_FALSE(
      logger_config_true.ShouldPrint(shape_inference_pass.get(), func_op));

  // Test with environment variable set to "FALSE".
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "FALSE", 1);
  BridgeLoggerConfig logger_config_false;
  EXPECT_TRUE(
      logger_config_false.ShouldPrint(shape_inference_pass.get(), func_op));

  // Test with environment variable set to "0".
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "0", 1);
  BridgeLoggerConfig logger_config_zero;
  EXPECT_TRUE(
      logger_config_zero.ShouldPrint(shape_inference_pass.get(), func_op));
}

// Test combinations of pass filter and string filter.
TEST_F(BridgeLoggerFilters, TestPassFilterAndStringFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // String filter is matched but pass filter is not.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER",
         "(tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>", 1);
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "ensorFlowShapeInferencePass", 1);
  BridgeLoggerConfig logger_config1;
  EXPECT_FALSE(logger_config1.ShouldPrint(shape_inference_pass.get(),
                                          mlir_module_with_add.get()));

  // Pass filter is matched but string filter is not.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "XXX", 1);
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "TensorFlowShapeInferencePass", 1);
  BridgeLoggerConfig logger_config2;
  EXPECT_FALSE(logger_config2.ShouldPrint(shape_inference_pass.get(),
                                          mlir_module_with_add.get()));

  // Both filters are matched.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER",
         "(tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>", 1);
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "TensorFlowShapeInferencePass", 1);
  BridgeLoggerConfig logger_config3;
  EXPECT_TRUE(logger_config3.ShouldPrint(shape_inference_pass.get(),
                                         mlir_module_with_add.get()));
}

// Test combinations of pass filter and enable only top level passes filter.
TEST_F(BridgeLoggerFilters, TestPassFilterAndEnableOnlyTopLevelPassesFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);

  // Deserialize the module with a Sub operation.
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_sub;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_sub, &mlir_context,
                                     &mlir_module_with_sub));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();
  std::unique_ptr<mlir::Pass> inliner_pass = mlir::createInlinerPass();

  // Find the nested function operation within the module.
  mlir::Operation *func_op = nullptr;
  for (auto &op : mlir_module_with_sub->getOps()) {
    if (llvm::isa<mlir::func::FuncOp>(&op)) {
      func_op = &op;
      break;
    }
  }
  ASSERT_NE(func_op, nullptr);

  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "TensorFlowShapeInferencePass", 1);
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "1", 1);

  // Create a BridgeLoggerConfig
  BridgeLoggerConfig logger_config;

  // ShouldPrint should return true for top-level operation with matching pass
  // filter.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_sub.get()));

  // ShouldPrint should return false for nested operation when
  // enable_only_top_level_passes_ is true.
  EXPECT_FALSE(logger_config.ShouldPrint(shape_inference_pass.get(), func_op));

  // ShouldPrint should return false for pass not matching the pass filter.
  EXPECT_FALSE(logger_config.ShouldPrint(inliner_pass.get(),
                                         mlir_module_with_sub.get()));

  // Unset enable_only_top_level_passes_, so enable_only_top_level_passes_
  // is false.
  unsetenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");
  BridgeLoggerConfig logger_config_no_suppress;

  // Now ShouldPrint should return true for nested operation
  EXPECT_TRUE(logger_config_no_suppress.ShouldPrint(shape_inference_pass.get(),
                                                    func_op));
}

// Test combinations of string filter and enable only top level passes filter.
TEST_F(BridgeLoggerFilters, TestStringFilterAndEnableOnlyTopLevelPassesFilter) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);

  // Deserialize the module with an Add operation.
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // Find the nested function operation within the module.
  mlir::Operation *func_op = nullptr;
  for (auto &op : mlir_module_with_add->getOps()) {
    if (llvm::isa<mlir::func::FuncOp>(&op)) {
      func_op = &op;
      break;
    }
  }
  ASSERT_NE(func_op, nullptr);

  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "tf.AddV2", 1);
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "1", 1);

  BridgeLoggerConfig logger_config;

  // ShouldPrint should return true for top-level operation containing
  // "tf.AddV2".
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));

  // ShouldPrint should return false for nested operation due to
  // enable_only_top_level_passes_.
  EXPECT_FALSE(logger_config.ShouldPrint(shape_inference_pass.get(), func_op));

  // Unset enable_only_top_level_passes_, so enable_only_top_level_passes_
  // is false.
  unsetenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");
  BridgeLoggerConfig logger_config_no_suppress;

  // Now ShouldPrint should return true for nested operation since string filter
  // matches.
  EXPECT_TRUE(logger_config_no_suppress.ShouldPrint(shape_inference_pass.get(),
                                                    func_op));

  // Change string filter to not match any operation.
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "NonExistentOp", 1);
  BridgeLoggerConfig logger_config_no_match;

  // ShouldPrint should return false since string filter does not match.
  EXPECT_FALSE(logger_config_no_match.ShouldPrint(shape_inference_pass.get(),
                                                  mlir_module_with_add.get()));
}

// Test combinations where all filters are set but none match.
TEST_F(BridgeLoggerFilters, TestAllFiltersNoMatch) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);

  // Deserialize the module with a Sub operation.
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_sub;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_sub, &mlir_context,
                                     &mlir_module_with_sub));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();

  // Set pass filter to not match any pass
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "NonExistentPass", 1);
  // Set string filter to not match any string
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "NonExistentOp", 1);
  // Set suppress_nested_passes_ to true
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "1", 1);

  BridgeLoggerConfig logger_config;

  // ShouldPrint should return false since none of the filters match.
  EXPECT_FALSE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                         mlir_module_with_sub.get()));

  // Unset enable_only_top_level_passes_.
  unsetenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");
  BridgeLoggerConfig logger_config_no_suppress;

  // ShouldPrint should still return false since pass and string filters do not
  // match.
  EXPECT_FALSE(logger_config_no_suppress.ShouldPrint(
      shape_inference_pass.get(), mlir_module_with_sub.get()));
}

// Test combinations of all three filters.
TEST_F(BridgeLoggerFilters, TestAllFiltersCombination) {
  mlir::DialectRegistry mlir_registry;
  mlir::RegisterAllTensorFlowDialects(mlir_registry);
  mlir::MLIRContext mlir_context(mlir_registry);

  // Deserialize the module with an Add operation.
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module_with_add;
  TF_ASSERT_OK(DeserializeMlirModule(module_with_add, &mlir_context,
                                     &mlir_module_with_add));

  std::unique_ptr<mlir::Pass> shape_inference_pass =
      mlir::TF::CreateTFShapeInferencePass();
  std::unique_ptr<mlir::Pass> inliner_pass = mlir::createInlinerPass();

  // Find the nested function operation within the module.
  mlir::Operation *func_op = nullptr;
  for (auto &op : mlir_module_with_add->getOps()) {
    if (llvm::isa<mlir::func::FuncOp>(&op)) {
      func_op = &op;
      break;
    }
  }
  ASSERT_NE(func_op, nullptr);

  // Set all three filters.
  setenv("MLIR_BRIDGE_LOG_PASS_FILTER", "TensorFlowShapeInferencePass", 1);
  setenv("MLIR_BRIDGE_LOG_STRING_FILTER", "tf.AddV2", 1);
  setenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES", "1", 1);

  // Create a BridgeLoggerConfig which will read the environment variables.
  BridgeLoggerConfig logger_config;

  // Test that ShouldPrint returns true if all filters pass and operation is
  // top-level.
  EXPECT_TRUE(logger_config.ShouldPrint(shape_inference_pass.get(),
                                        mlir_module_with_add.get()));

  // Test that ShouldPrint returns false if suppress_nested_passes_ is true and
  // operation is nested.
  EXPECT_FALSE(logger_config.ShouldPrint(shape_inference_pass.get(), func_op));

  // Change to a pass that does not match the pass filter.
  EXPECT_FALSE(logger_config.ShouldPrint(inliner_pass.get(),
                                         mlir_module_with_add.get()));

  // Unset enable_only_top_level_passes_ to false.
  unsetenv("MLIR_BRIDGE_LOG_ENABLE_ONLY_TOP_LEVEL_PASSES");

  BridgeLoggerConfig logger_config_no_suppress;

  // Now ShouldPrint should return true for nested operation since
  // enable_only_top_level_passes_ is false.
  EXPECT_TRUE(logger_config_no_suppress.ShouldPrint(shape_inference_pass.get(),
                                                    func_op));

  // Change to a pass that does not match the pass filter.
  EXPECT_FALSE(logger_config_no_suppress.ShouldPrint(
      inliner_pass.get(), mlir_module_with_add.get()));
}
}  // namespace
}  // namespace tensorflow
