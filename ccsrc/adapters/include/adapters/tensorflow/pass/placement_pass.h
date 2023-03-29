#pragma once
#ifndef FRAMEWORK_ADAPTERS_TENSORFLOW_PLACEMENT_PASS_H
#define FRAMEWORK_ADAPTERS_TENSORFLOW_PLACEMENT_PASS_H
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
class PlacementPass : public GraphOptimizationPass {
  public:
    explicit PlacementPass() {}

    Status Run(const GraphOptimizationPassOptions& options) override;
};
}  // namespace tensorflow

#endif /* end of include guard: PLACEMENT_PASS_H */
