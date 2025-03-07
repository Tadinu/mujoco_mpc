#pragma once

#include <random>

namespace mjpc {
struct Random {
  static int seed;
  // PRN, random seed for the random number engine
  static std::random_device rd;

  // Standard mersenne_twister_engine seeded with rd()
  static std::mt19937_64 gen;

  template <typename T>
  static T rand(const T min, const T max) {
    if constexpr (std::is_integral_v<T>) {
      return std::uniform_int_distribution<T>(min, max)(gen);
    } else if constexpr (std::is_floating_point_v<T>) {
      return std::uniform_real_distribution<T>(min, max)(gen);
    } else {
      return {};
    }
  }

  static double rand() { return rand<double>(0.f, 1.f); }
  static bool rand_bool() { return rand() < 0.5f; }
};
}
