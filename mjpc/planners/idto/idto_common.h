#pragma once

using IdtoSharedMutexLock = std::shared_lock<std::shared_mutex>;
using IdtoMutexLock = std::lock_guard<std::mutex>;

using DrakeMeshcatPtr = std::shared_ptr<drake::geometry::Meshcat>;

#define IDTO_USE_MESHCAT (1)