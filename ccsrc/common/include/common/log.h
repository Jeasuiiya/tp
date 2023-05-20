#ifndef FRAMEWORK_COMMON_LOG_H
#define FRAMEWORK_COMMON_LOG_H
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "spdlog/cfg/env.h"  // support for loading levels from the environment variable
#include "spdlog/spdlog.h"
namespace framework::log {
inline void Init() {
    spdlog::cfg::load_env_levels();
}
struct LogInit {
    LogInit() {
        Init();
    }
};
static LogInit log_init;
}  // namespace framework::log

#endif
