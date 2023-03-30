#include "policy/aware/aware_interface.h"

namespace framework {
bool AwareInterface::StartReinLearningModule() {
    return py_aware_interface.attr("startup_strategy")().cast<bool>();
}

void AwareInterface::GetReinLearningBestPlacement(std::map<std::string, std::string>* best_placement) {
    py::dict py_best_placement = py_aware_interface.attr("get_best_placement")();
    for (const auto& node_to_device : py_best_placement) {
        auto node = node_to_device.first.cast<std::string>();
        auto device_id = node_to_device.second.cast<int64_t>();
        std::string device = "/GPU:" + std::to_string(device_id);
        best_placement->insert(std::pair<std::string, std::string>(node, device));
    }
}

}  // namespace framework
