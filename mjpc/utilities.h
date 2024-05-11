// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_UTILITIES_H_
#define MJPC_UTILITIES_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

#include <absl/container/flat_hash_map.h>
#include <absl/log/check.h>
#include <absl/strings/match.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include <mujoco/mujoco.h>

#include "mjpc/array_safety.h"

namespace mjpc {

// maximum number of traces that are visualized
inline constexpr int kMaxTraces = 99;

// set mjData state
void SetState(const mjModel* model, mjData* data, const double* state);

// get mjData state
void GetState(const mjModel* model, const mjData* data, double* state);

// 2d vector dot-product
inline mjtNum mju_dot2(const mjtNum vec1[2], const mjtNum vec2[2]) {
  return vec1[0] * vec2[0] + vec1[1] * vec2[1];
}

// 2d vector squared distance
inline mjtNum mju_sqrdist2(const mjtNum vec1[2], const mjtNum vec2[2]) {
  const mjtNum diff[2] = {vec1[0] - vec2[0], vec1[1] - vec2[1]};
  return mju_dot2(diff, diff);
}

// get numerical data from a custom element in mjModel with the given name
inline double* GetCustomNumericData(const mjModel* m, std::string_view name) {
  for (int i = 0; i < m->nnumeric; i++) {
    if (std::string_view(m->names + m->name_numericadr[i]) == name) {
      return m->numeric_data + m->numeric_adr[i];
    }
  }
  return nullptr;
}

// get text data from a custom element in mjModel with the given name
char* GetCustomTextData(const mjModel* m, std::string_view name);

// get a scalar value from a custom element in mjModel with the given name
template <typename T>
std::optional<T> GetNumber(const mjModel* m, std::string_view name) {
  double* data = GetCustomNumericData(m, name);
  if (data) {
    return static_cast<T>(data[0]);
  } else {
    return std::nullopt;
  }
}

// get a single numerical value from a custom element in mjModel, or return the
// default value if a custom element with the specified name does not exist
template <typename T>
T GetNumberOrDefault(T default_value, const mjModel* m, std::string_view name) {
  return GetNumber<T>(m, name).value_or(default_value);
}

// reinterpret double as int
inline int ReinterpretAsInt(double value) {
  return *std::launder(reinterpret_cast<const int*>(&value));
}

inline double ReinterpretAsDouble(int64_t value) {
  return *std::launder(reinterpret_cast<const double*>(&value));
}

// returns a map from custom field name to the list of valid values for that
// field
absl::flat_hash_map<std::string, std::vector<std::string>>
ResidualSelectionLists(const mjModel* m);

// get the string selected in a drop down with the given name, given the value
// in the residual parameters vector
std::string ResidualSelection(const mjModel* m, std::string_view name,
                              double residual_parameter);
// returns a value for residual parameters that fits the given text value
// in the given list
double ResidualParameterFromSelection(const mjModel* m, std::string_view name,
                                      std::string_view value);

// returns a default value to put in residual parameters, given the index of a
// custom numeric attribute in the model
inline double DefaultResidualSelection(const mjModel* m, int numeric_index) {
  // list selections are stored as ints, but numeric values are doubles.
  int64_t value = m->numeric_data[m->numeric_adr[numeric_index]];
  return *std::launder(reinterpret_cast<const double*>(&value));
}
// Clamp x between bounds, e.g., bounds[0] <= x[i] <= bounds[1]
inline void Clamp(double* x, const double* bounds, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = mju_clip(x[i], bounds[2 * i], bounds[2 * i + 1]);
  }
}

// get sensor data using string
inline double* SensorByName(const mjModel* m, const mjData* d,
                            const char* name) {
  int id = mj_name2id(m, mjOBJ_SENSOR, name);
  if (id == -1) {
    std::cerr << "sensor \"" << std::string(name) << "\" not found.\n";
    return nullptr;
  } else {
    return d->sensordata + m->sensor_adr[id];
  }
}

double DefaultParameterValue(const mjModel* model, std::string_view name);

// get index to residual parameter data using string
inline int ParameterIndex(const mjModel* model, std::string_view name) {
  int id =
      mj_name2id(model, mjOBJ_NUMERIC, absl::StrCat("residual_", name).c_str());

  if (id == -1) {
    mju_error_s("Parameter '%s' not found", std::string(name).c_str());
  }

  int i;
  for (i = 0; i < model->nnumeric; i++) {
    const char* first_residual = mj_id2name(model, mjOBJ_NUMERIC, i);
    if (absl::StartsWith(first_residual, "residual_")) {
      break;
    }
  }
  return id - i;
}

inline int CostTermByName(const mjModel* m, const std::string& name) {
  int id = mj_name2id(m, mjOBJ_SENSOR, name.c_str());
  if (id == -1 || m->sensor_type[id] != mjSENS_USER) {
    std::cerr << "cost term \"" << name << "\" not found.\n";
    return -1;
  } else {
    return id;
  }
}
// return total size of sensors of type user
inline int ResidualSize(const mjModel* model) {
  int user_sensor_dim = 0;
  bool encountered_nonuser_sensor = false;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
      if (encountered_nonuser_sensor) {
        mju_error("user type sensors must come before other sensor types");
      }
    } else {
      encountered_nonuser_sensor = true;
    }
  }
  return user_sensor_dim;
}

// sanity check that residual size equals total user-sensor dimension
inline void CheckSensorDim(const mjModel* model, int residual_size) {
  int user_sensor_dim = ResidualSize(model);
  if (user_sensor_dim != residual_size) {
    mju_error(
        "mismatch between total user-sensor dimension %d "
        "and residual size %d",
        user_sensor_dim, residual_size);
  }
}

// get traces from sensors
void GetTraces(double* traces, const mjModel* m, const mjData* d,
               int num_trace);

// get keyframe `qpos` data using string
inline double* KeyQPosByName(const mjModel* m, const mjData* d,
                      const std::string& name) {
  int id = mj_name2id(m, mjOBJ_KEY, name.c_str());
  if (id == -1) {
    return nullptr;
  }
  return m->key_qpos + m->nq * id;
}

// fills t with N numbers, starting from t0 and incrementing by t_step
void LinearRange(double* t, double t_step, double t0, int N);

// find interval in monotonic sequence containing value
template <typename T>
void FindInterval(int* bounds, const std::vector<T>& sequence, double value,
                  int length) {
  // get bounds
  auto it =
      std::upper_bound(sequence.begin(), sequence.begin() + length, value);
  int upper_bound = it - sequence.begin();
  int lower_bound = upper_bound - 1;

  // set bounds
  if (lower_bound < 0) {
    bounds[0] = 0;
    bounds[1] = 0;
  } else if (lower_bound > length - 1) {
    bounds[0] = length - 1;
    bounds[1] = length - 1;
  } else {
    bounds[0] = mju_max(lower_bound, 0);
    bounds[1] = mju_min(upper_bound, length - 1);
  }
}

// zero-order interpolation
void ZeroInterpolation(double* output, double x, const std::vector<double>& xs,
                       const double* ys, int dim, int length);

// linear interpolation
void LinearInterpolation(double* output, double x,
                         const std::vector<double>& xs, const double* ys,
                         int dim, int length);

// coefficients for cubic interpolation
void CubicCoefficients(double* coefficients, double x,
                       const std::vector<double>& xs, int T);

// finite-difference vector
double FiniteDifferenceSlope(double x, const std::vector<double>& xs,
                             const double* ys, int dim, int length, int i);

// cubic polynomial interpolation
void CubicInterpolation(double* output, double x, const std::vector<double>& xs,
                        const double* ys, int dim, int length);

// returns the path to the directory containing the current executable
inline std::string GetExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError()
                  << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#else
  const char* path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: "
                  << std::strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// Returns the directory where tasks are stored
inline std::string GetTasksDir() {
  const char* tasks_dir = std::getenv("MJPC_TASKS_DIR");
  if (tasks_dir) {
    return tasks_dir;
  }
  return GetExecutableDir() + "/../mjpc/tasks";
}

// returns path to a model XML file given path relative to models dir
inline std::string GetModelPath(std::string_view path)
{
  return GetTasksDir() + "/" + std::string(path);
}

// dx = (x2 - x1) / h
void Diff(mjtNum* dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n);

// finite-difference two state vectors ds = (s2 - s1) / h
void StateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2,
               mjtNum h);

// return global height of nearest geom in geomgroup under given position
// return global height of nearest group 0 geom under given position
inline mjtNum Ground(const mjModel* model, const mjData* data, const mjtNum pos[3],
              const mjtByte* geomgroup = nullptr) {
  mjtNum down[3] = {0, 0, -1};                      // aim ray straight down
  const mjtNum height_offset = .5;  // add some height in case of penetration
  const mjtByte flg_static = 1;     // include static geoms
  const int bodyexclude = -1;       // don't exclude any bodies
  int geomid;                       // id of intersecting geom
  mjtNum query[3] = {pos[0], pos[1], pos[2] + height_offset};
  const mjtByte default_geomgroup[6] = {1, 0, 0, 0, 0, 0};
  const mjtByte* query_geomgroup = geomgroup ? geomgroup : default_geomgroup;
  mjtNum dist = mj_ray(model, data, query, down, query_geomgroup, flg_static,
                       bodyexclude, &geomid);

  if (dist < 0) {  // SHOULD NOT OCCUR
    mju_error("no group 0 geom detected by raycast");
  }

  return pos[2] + height_offset - dist;
}

// set x to be the point on the segment [p0 p1] that is nearest to x
void ProjectToSegment(double x[3], const double p0[3], const double p1[3]);

// find frame that best matches 4 feet, z points to body
void FootFrame(double feet_pos[3], double feet_mat[9], double feet_quat[4],
               const double body[3], const double foot0[3],
               const double foot1[3], const double foot2[3],
               const double foot3[3]);

// default cost colors
static const float CostColors[20][3] {
    {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.3, 0.3, 1.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 1.0}, {1.0, 0.5, 0.5}, {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0},
    {1.0, 1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
    {0.3, 0.3, 1.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.5, 0.5},
    {0.5, 1.0, 0.5}, {0.5, 0.5, 1.0}, {1.0, 1.0, 0.5}, {0.5, 1.0, 1.0},
};

constexpr int kNCostColors = sizeof(CostColors) / (sizeof(float) * 3);

// plots - vertical line
void PlotVertical(mjvFigure* fig, double time, double min_value,
                  double max_value, int N, int index);

// plots - update data
inline void PlotUpdateData(mjvFigure* fig, double* bounds, double x, double y,
                    int length, int index, int x_update, int y_update,
                    double x_bound_lower) {
  int pnt = mjMIN(length, fig->linepnt[index] + 1);

  // shift previous data
  for (int i = pnt - 1; i > 0; i--) {
    if (x_update) {
      fig->linedata[index][2 * i] = fig->linedata[index][2 * i - 2];
    }
    if (y_update) {
      fig->linedata[index][2 * i + 1] = fig->linedata[index][2 * i - 1];
    }

    // bounds
    if (fig->linedata[index][2 * i] > x_bound_lower) {
      if (fig->linedata[index][2 * i + 1] < bounds[0]) {
        bounds[0] = fig->linedata[index][2 * i + 1];
      }
      if (fig->linedata[index][2 * i + 1] > bounds[1]) {
        bounds[1] = fig->linedata[index][2 * i + 1];
      }
    }
  }

  // current data
  fig->linedata[index][0] = x;
  fig->linedata[index][1] = y;
  fig->linepnt[index] = pnt;
}

// plots - reset
void PlotResetData(mjvFigure* fig, int length, int index);

// plots - horizontal line
void PlotHorizontal(mjvFigure* fig, const double* xs, double y, int length,
                    int index);

// plots - set data
void PlotData(mjvFigure* fig, double* bounds, const double* xs,
              const double* ys, int dim, int dim_limit, int length,
              int start_index, double x_bound_lower);

// add geom to scene
// add geom to scene
inline void AddGeom(mjvScene* scene, mjtGeom type, const mjtNum size[3],
             const mjtNum pos[3], const mjtNum mat[9], const float rgba[4]) {
  // if no available geoms, return
  if (scene->ngeom >= scene->maxgeom) return;

  // add geom
  mjtNum mat_world[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjv_initGeom(&scene->geoms[scene->ngeom], type, size, pos,
               mat ? mat : mat_world, rgba);
  scene->geoms[scene->ngeom].category = mjCAT_DECOR;

  // increment ngeom
  scene->ngeom += 1;
}

// add connector geom to scene
// add connector geom to scene
inline void AddConnector(mjvScene* scene, mjtGeom type, mjtNum width,
                  const mjtNum from[3], const mjtNum to[3],
                  const float rgba[4]) {
  // if no available geoms, return
  if (scene->ngeom >= scene->maxgeom) return;

  // make connector geom
  mjv_initGeom(&scene->geoms[scene->ngeom], type,
               /*size=*/nullptr, /*pos=*/nullptr, /*mat=*/nullptr, rgba);
  scene->geoms[scene->ngeom].category = mjCAT_DECOR;
  mjv_makeConnector(&scene->geoms[scene->ngeom], type, width, from[0], from[1],
                    from[2], to[0], to[1], to[2]);

  // increment ngeom
  scene->ngeom += 1;
}

// number of available hardware threads
// number of available hardware threads
#if defined(__APPLE__) || defined(_WIN32)
inline int NumAvailableHardwareThreads(void) {
  return std::thread::hardware_concurrency();
}
#else
inline int NumAvailableHardwareThreads(void) {
  // start by assuming a maximum of 128 hardware threads and keep growing until
  // the cpu_set_t is big enough to hold the mask for the entire machine
  for (int max_count = 128; true; max_count *= 2) {
    std::unique_ptr<cpu_set_t, void (*)(cpu_set_t*)> set(
        CPU_ALLOC(max_count), +[](cpu_set_t* set) { CPU_FREE(set); });
    size_t setsize = CPU_ALLOC_SIZE(max_count);
    int result = sched_getaffinity(getpid(), setsize, set.get());
    if (result == 0) {
      // success
      return CPU_COUNT_S(setsize, set.get());
    } else if (errno != EINVAL) {
      // failure other than max_count being too small, just return 1
      return 1;
    }
  }
}
#endif

// check mjData for warnings, return true if any warnings
bool CheckWarnings(mjData* data);

// compute vector with log-based scaling between min and max values
void LogScale(double* values, double max_value, double min_value, int steps);

// get a pointer to a specific element of a vector, or nullptr if out of bounds
template <typename T>
inline T* DataAt(std::vector<T>& vec, typename std::vector<T>::size_type elem) {
  if (elem < vec.size()) {
    return &vec[elem];
  } else {
    return nullptr;
  }
}

// increases the value of an atomic variable.
// in C++20 atomic::operator+= is built-in for floating point numbers, but this
// function works in C++11
inline void IncrementAtomic(std::atomic<double>& v, double a) {
  for (double t = v.load(); !v.compare_exchange_weak(t, t + a);) {
  }
}

// get a pointer to a specific element of a vector, or nullptr if out of bounds
template <typename T>
inline const T* DataAt(const std::vector<T>& vec,
                       typename std::vector<T>::size_type elem) {
  return DataAt(const_cast<std::vector<T>&>(vec), elem);
}

using UniqueMjData = std::unique_ptr<mjData, void (*)(mjData*)>;

inline UniqueMjData MakeUniqueMjData(mjData* d) {
  return UniqueMjData(d, mj_deleteData);
}

using UniqueMjModel = std::unique_ptr<mjModel, void (*)(mjModel*)>;

inline UniqueMjModel MakeUniqueMjModel(mjModel* d) {
  return UniqueMjModel(d, mj_deleteModel);
}

// returns 2D point on line segment from v0 to v1 that is nearest to query point
inline void ProjectToSegment2D(mjtNum res[2], const mjtNum query[2],
                        const mjtNum v0[2], const mjtNum v1[2]) {
  mjtNum axis[2] = {v1[0] - v0[0], v1[1] - v0[1]};
  mjtNum length = mju_sqrt(mju_dot2(axis, axis));
  axis[0] /= length;
  axis[1] /= length;
  mjtNum center[2] = {0.5 * (v1[0] + v0[0]), 0.5 * (v1[1] + v0[1])};
  mjtNum t = mju_dot2(query, axis) - mju_dot2(center, axis);
  t = mju_clip(t, -length / 2, length / 2);
  res[0] = center[0] + t * axis[0];
  res[1] = center[1] + t * axis[1];
}

// returns point in 2D convex hull that is nearest to query
inline void NearestInHull(mjtNum res[2], const mjtNum query[2], const mjtNum* points,
                   const int* hull, int num_hull) {
  int outside = 0;      // assume query point is inside the hull
  mjtNum best_sqrdist;  // smallest squared distance so far
  for (int i = 0; i < num_hull; i++) {
    const mjtNum* v0 = points + 2 * hull[i];
    const mjtNum* v1 = points + 2 * hull[(i + 1) % num_hull];

    // edge from v0 to v1
    mjtNum e01[2] = {v1[0] - v0[0], v1[1] - v0[1]};

    // normal to the edge, pointing *into* the convex hull
    mjtNum n01[2] = {-e01[1], e01[0]};

    // if constraint is active, project to the edge, compare to best so far
    mjtNum v0_to_query[2] = {query[0] - v0[0], query[1] - v0[1]};
    if (mju_dot(v0_to_query, n01, 2) < 0) {
      mjtNum projection[2];
      ProjectToSegment2D(projection, query, v0, v1);
      mjtNum sqrdist = mju_sqrdist2(projection, query);
      if (!outside || (outside && sqrdist < best_sqrdist)) {
        // first or closer candidate, copy to res
        res[0] = projection[0];
        res[1] = projection[1];
        best_sqrdist = sqrdist;
      }
      outside = 1;
    }
  }

  // not outside any edge, return the query point
  if (!outside) {
    res[0] = query[0];
    res[1] = query[1];
  }
}

// returns true if edge to candidate is to the right of edge to next
inline bool IsEdgeOutside(const mjtNum current[2], const mjtNum next[2],
                   const mjtNum candidate[2]) {
  mjtNum current_edge[2] = {next[0] - current[0], next[1] - current[1]};
  mjtNum candidate_edge[2] = {candidate[0] - current[0],
                              candidate[1] - current[1]};
  mjtNum rotated_edge[2] = {current_edge[1], -current_edge[0]};
  mjtNum projection = mju_dot2(candidate_edge, rotated_edge);

  // check if candidate edge is to the right
  if (projection > mjMINVAL) {
    // actually to the right: accept
    return true;
  } else if (abs(projection) < mjMINVAL) {
    // numerically equivalent: accept if longer
    mjtNum current_length2 = mju_dot2(current_edge, current_edge);
    mjtNum candidate_length2 = mju_dot2(candidate_edge, candidate_edge);
    return (candidate_length2 > current_length2);
  }
  // not to the right
  return false;
}

// find the convex hull of a small set of 2D points
inline int Hull2D(int* hull, int num_points, const mjtNum* points) {
  // handle small number of points
  if (num_points < 1) return 0;
  hull[0] = 0;
  if (num_points == 1) return 1;
  if (num_points == 2) {
    hull[1] = 1;
    return 2;
  }

  // find the point with largest x value - must lie on hull
  mjtNum best_x = points[0];
  mjtNum best_y = points[1];
  for (int i = 1; i < num_points; i++) {
    mjtNum x = points[2 * i];
    mjtNum y = points[2 * i + 1];

    // accept if larger, use y value to tie-break exact equality
    if (x > best_x || (x == best_x && y > best_y)) {
      best_x = x;
      best_y = y;
      hull[0] = i;
    }
  }

  //  Gift-wrapping algorithm takes time O(nh)
  // TODO(benmoran) Investigate faster convex hull methods.
  int num_hull = 1;
  for (int i = 0; i < num_points; i++) {
    // loop over all points, find point that is furthest outside
    int next = -1;
    for (int candidate = 0; candidate < num_points; candidate++) {
      if ((next == -1) ||
          IsEdgeOutside(points + 2 * hull[num_hull - 1], points + 2 * next,
                        points + 2 * candidate)) {
        next = candidate;
      }
    }

    // termination condition
    if ((num_hull > 1) && (next == hull[0])) {
      break;
    }

    // add new point
    hull[num_hull++] = next;
  }

  return num_hull;
}

// TODO(etom): move findiff-related functions to a different library.

// finite-difference gradient
class FiniteDifferenceGradient {
 public:
  // constructor
  explicit FiniteDifferenceGradient(int dim);

  // resize memory
  void Resize(int dim);

  // compute gradient
  void Compute(std::function<double(const double* x)> func,
                  const double* input, int dim);

  // members
  std::vector<double> gradient;
  double epsilon = 1.0e-5;
 private:
  std::vector<double> workspace_;
};

// finite-difference Jacobian
class FiniteDifferenceJacobian {
 public:
  // constructor
  FiniteDifferenceJacobian(int num_output, int num_input);

  // resize memory
  void Resize(int num_output, int num_input);

  // compute Jacobian
  void Compute(std::function<void(double* output, const double* input)> func,
                  const double* input, int num_output, int num_input);

  // members
  std::vector<double> jacobian;
  std::vector<double> jacobian_transpose;
  std::vector<double> output;
  std::vector<double> output_nominal;
  double epsilon = 1.0e-5;
 private:
  std::vector<double> workspace_;
};

// finite-difference Hessian
class FiniteDifferenceHessian {
 public:
  // constructor
  explicit FiniteDifferenceHessian(int dim);

  // resize memory
  void Resize(int dim);

  // compute
  void Compute(std::function<double(const double* x)> func,
                  const double* input, int dim);

  // members
  std::vector<double> hessian;
  double epsilon = 1.0e-5;
 private:
  std::vector<double> workspace1_;
  std::vector<double> workspace2_;
  std::vector<double> workspace3_;
};

// set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void SetBlockInMatrix(double* mat, const double* block, double scale, int rm,
                      int cm, int rb, int cb, int ri, int ci);

// set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void AddBlockInMatrix(double* mat, const double* block, double scale, int rm,
                      int cm, int rb, int cb, int ri, int ci);

// get block (size: rb x cb) from mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void BlockFromMatrix(double* block, const double* mat, int rb, int cb, int rm,
                     int cm, int ri, int ci);

// differentiate mju_subQuat wrt qa, qb
void DifferentiateSubQuat(double jaca[9], double jacb[9], const double qa[4],
                          const double qb[4]);

// differentiate velocity by finite-differencing two positions wrt to qpos1,
// qpos2
void DifferentiateDifferentiatePos(double* jac1, double* jac2,
                                   const mjModel* model, double dt,
                                   const double* qpos1, const double* qpos2);

// compute number of nonzeros in band matrix
int BandMatrixNonZeros(int ntotal, int nband);

// TODO(etom): rename (SecondsSince?)
// get duration since time point
inline double GetDuration(std::chrono::steady_clock::time_point time) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - time)
      .count();
}

// copy symmetric band matrix block by block
void SymmetricBandMatrixCopy(double* res, const double* mat, int dblock,
                             int nblock, int ntotal, int num_blocks,
                             int res_start_row, int res_start_col,
                             int mat_start_row, int mat_start_col,
                             double* scratch);

// zero block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void ZeroBlockInMatrix(double* mat, int rm, int cm, int rb, int cb, int ri,
                       int ci);

// square dense to block band matrix
void DenseToBlockBand(double* res, int dim, int dblock, int nblock);

// infinity norm
template <typename T>
T InfinityNorm(T* x, int n) {
  return std::abs(*std::max_element(x, x + n, [](T a, T b) -> bool {
    return (std::abs(a) < std::abs(b));
  }));
}

// trace of square matrix
double Trace(const double* mat, int n);

// determinant of 3x3 matrix
double Determinant3(const double* mat);

// inverse of 3x3 matrix
void Inverse3(double* res, const double* mat);

// condition matrix: res = mat11 - mat10 * mat00 \ mat10^T; return rank of mat00
// TODO(taylor): thread
void ConditionMatrix(double* res, const double* mat, double* mat00,
                     double* mat10, double* mat11, double* tmp0, double* tmp1,
                     int n, int n0, int n1, double* bandfactor = NULL,
                     int nband = 0);

// principal eigenvector of 4x4 matrix
// QUEST algorithm from "Three-Axis Attitude Determination from Vector
// Observations"
void PrincipalEigenVector4(double* res, const double* mat,
                           double eigenvalue_init = 12.0);

// set scaled symmetric block matrix in band matrix
void SetBlockInBand(double* band, const double* block, double scale, int ntotal,
                    int nband, int nblock, int shift, int row_skip = 0,
                    bool add = true);

}  // namespace mjpc

#endif  // MJPC_UTILITIES_H_
