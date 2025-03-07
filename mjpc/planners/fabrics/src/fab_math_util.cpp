#include "mjpc/planners/fabrics/include/fab_math_util.h"

#include "mjpc/utils/mjpc_math_util.h"
#include "mjpc/planners/fabrics/include/fab_common.h"

int mjpc::Random::seed = 1000;
std::random_device mjpc::Random::rd;

// Standard mersenne_twister_engine seeded with rd()
#if FAB_RANDOM_DETERMINISTIC
std::mt19937_64 FabRandomGenerator::gen(seed);
#else
std::mt19937_64 mjpc::Random::gen(rd());
#endif