#include <policy/trinity/trinity_interface.h>

namespace framework {
bool TrinityInterface::StartReinLearningModule() {
    return py_trinity_interface.attr("startup_strategy")().cast<bool>();
    ;
}

void TrinityInterface::GetReinLearningBestPlacement(std::map<std::string, std::string>* best_placement) {
    py::dict py_best_placement = py_trinity_interface.attr("get_best_placement")();
    for (const auto& node_to_device : py_best_placement) {
        auto node = node_to_device.first.cast<std::string>();
        auto device = node_to_device.second.cast<std::string>();
        best_placement->insert(std::pair<std::string, std::string>(node, device));
    }
}

}  // namespace framework