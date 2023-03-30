#ifndef FRAMEWORK_COMMON_LOG_H
#define FRAMEWORK_COMMON_LOG_H
#include "spdlog/cfg/env.h"  // support for loading levels from the environment variable
#include "spdlog/spdlog.h"

namespace framework::log {
inline void Init() {
    spdlog::cfg::load_env_levels();
}
}  // namespace framework::log

#endif
