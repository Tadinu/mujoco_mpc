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

#include <absl/container/flat_hash_map.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

// Abseil
#include "absl/container/btree_set.h"
#include "absl/types/span.h"

// mujoco
#include <mujoco/mujoco.h>
#include "mjpc/utils/mjpc_core_util.h"

#define MJPC_OPENMP_ENABLED (1)
#define MJPC_OPENMP_THREADS_NUM (1000)

namespace mjpc {
// maximum number of traces that are visualized
inline constexpr int kMaxTraces = 99;
inline constexpr mjtNum TRANSLATION_ZERO[3] = {0, 0, 0};
inline constexpr mjtNum ROTATION_IDENTITY[4] = {1, 0, 0, 0};
inline constexpr mjtNum POSE_IDENTITY[7] = {0, 0, 0, 1, 0, 0, 0};

// make model differentiable by setting solimp[0] to zero
void MakeDifferentiable(mjModel* model);

inline bool IsSelectedControl(const std::vector<int>& indices, int idx) {
  return indices.empty() || std::find(indices.begin(), indices.end(), idx) != indices.end();
}

// Joint
inline int QueryJointId(const mjModel* model, const char* joint_name) {
  return model ? mj_name2id(model, mjOBJ_JOINT, joint_name) : -1;
}

inline int QueryJointPosAddress(const mjModel* model, const char* joint_name) {
  int joint_id = QueryJointId(model, joint_name);
  return (model && (joint_id >= 0) && (joint_id < model->njnt)) ? model->jnt_qposadr[joint_id] : 0;
}

inline int QueryJointDofAddress(const mjModel* model, const char* joint_name) {
  int joint_id = QueryJointId(model, joint_name);
  return (model && (joint_id >= 0) && (joint_id < model->njnt)) ? model->jnt_dofadr[joint_id] : 0;
}

// Dof
inline int QueryDofId(const mjModel* model, const char* dof_name) {
  return model ? mj_name2id(model, mjOBJ_DOF, dof_name) : -1;
}

// NOTE: model_->nq,nv are actuated joints/controls configured in MJ model
// dof: full dof of the robot
inline std::vector<double> QueryJointPos(const mjModel* model, const mjData* data, int dof,
                                         const std::string& first_joint_name) {
  if (model && data) {
    std::vector<double> qpos(dof, 0);
    mju_copy(qpos.data(), data->qpos + QueryJointPosAddress(model, first_joint_name.c_str()),
             std::min(model->nq, dof));
    return qpos;
  }
  return {};
}

inline std::vector<double> QueryJointVel(const mjModel* model, const mjData* data, int dof,
                                         const std::string& first_joint_name) {
  if (model && data) {
    std::vector<double> qvel(dof, 0);
    mju_copy(qvel.data(), data->qvel + QueryJointDofAddress(model, first_joint_name.c_str()),
             std::min(model->nv, dof));
    return qvel;
  }
  return {};
}

// Body
inline int QueryBodyId(const mjModel* model, const char* body_name) {
  return model ? mj_name2id(model, mjOBJ_BODY, body_name) : -1;
}

inline mjtNum* QueryBodyQuat(const mjData* data, int body_id, bool inertia_com = true) {
  if (data) {
    if (inertia_com) {
      static mjtNum quat[4];
      mju_mat2Quat(quat, &data->ximat[9 * body_id]);
      return &quat[0];
    } else {
      return &data->xquat[4 * body_id];
    }
  }
  return nullptr;
}

inline mjtNum* QueryBodyRotMat(const mjData* data, int body_id, bool inertia_com = true) {
  if (data) {
    if (inertia_com) {
      return &data->ximat[9 * body_id];
    } else {
      static mjtNum mat[9];
      mju_quat2Mat(mat, &data->xquat[4 * body_id]);
      return &mat[0];
    }
  }
  return nullptr;
}

inline mjtNum* QueryBodyPos(const mjData* data, int body_id, bool inertia_com = true) {
  if (data) {
    return inertia_com ? &data->xipos[3 * body_id] : &data->xpos[3 * body_id];
  }
  return nullptr;
}

static std::pair<Eigen::Vector3d, Eigen::Quaterniond> QueryBodyPose(const mjModel* model, const mjData* data,
                                                                    const std::string& child_body_name,
                                                                    const std::string& parent_body_name =
                                                                        {}) {
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> res;
  mjtNum parent_pose[7];
  mju_copy(parent_pose, POSE_IDENTITY, 7);
  if (!parent_body_name.empty()) {
    auto parent_body_id = QueryBodyId(model, parent_body_name.c_str());
    mju_copy3(&parent_pose[0], QueryBodyPos(data, parent_body_id));
    mju_copy3(&parent_pose[3], QueryBodyQuat(data, parent_body_id));
  }

  mjtNum rel_pose[7];
  mju_copy(rel_pose, POSE_IDENTITY, 7);
  auto child_body_id = QueryBodyId(model, child_body_name.c_str());
  mjpc_localpos(&rel_pose[0], QueryBodyPos(data, child_body_id), &parent_pose[0],
               &parent_pose[3]);
  mjpc_localquat(&rel_pose[3], QueryBodyQuat(data, child_body_id), &parent_pose[3]);

  mju_copy3(res.first.data(), &rel_pose[0]);
#if 0
  res.second = Quaterniond(link_pose[3], // w
                           link_pose[4], // x
                           link_pose[5], // y
                           link_pose[6]  // z
      );
#else
  res.second = Eigen::Quaterniond(&rel_pose[3]);
#endif
  return res;
}

inline mjtNum* QueryBodyVel(const mjData* data, int body_id, bool linear = true) {
  if (data) {
    static double lvel[3] = {0};
#if 1
    mju_copy3(lvel, linear ? &data->cvel[6 * body_id + 3] : &data->cvel[6 * body_id]);
#else
    mjtNum vel[6];
    mj_objectVelocity(model_, data_, mjOBJ_BODY, body_id, vel, 0);
    mju_copy3(lvel, linear ? &vel[3] : &vel[0]);
#endif
    return &lvel[0];
  }
  return nullptr;
}

inline mjtNum* QueryBodyAcc(const mjData* data, int body_id, bool linear = true) {
  if (data) {
    static double lacc[3] = {0};
#if 1
    mju_copy3(lacc, linear ? &data->cacc[6 * body_id + 3] : &data->cacc[6 * body_id]);
#else
    mjtNum acc[6];
    mj_objectAcceleration(model_, data_, mjOBJ_BODY, body_id, acc, 0);
    mju_copy3(lacc, linear ? &acc[3] : &acc[0]);
#endif
    return &lacc[0];
  }
  return nullptr;
}

// Body mocap
inline int QueryBodyMocapId(const mjModel* model, const char* body_name) {
  if (model) {
    int body_id = QueryBodyId(model, body_name);
    return (body_id > -1) ? model->body_mocapid[body_id] : -1;
  }
  return -1;
}

inline void SetBodyMocapPos(const mjModel* model, const mjData* data, const char* body_name,
                            const double* pos) {
  if (data) {
    int bodyMocapId = QueryBodyMocapId(model, body_name);
    mju_copy3(&data->mocap_pos[3 * bodyMocapId], pos);
    //  std::cout << body_name << ":" << bodyMocapId << " " << pos[0] << " " << pos[1] << std::endl;
  }
}

inline mjtNum* QueryBodyMocapPos(const mjModel* model, const mjData* data, const char* body_name) {
  if (data) {
    int bodyMocapId = QueryBodyMocapId(model, body_name);
    return &data->mocap_pos[3 * bodyMocapId];
  }
  return nullptr;
}

inline void SetBodyMocapQuat(const mjModel* model, const mjData* data, const char* body_name,
                             const double* quat) {
  if (data) {
    int bodyMocapId = QueryBodyMocapId(model, body_name);
    mju_copy3(&data->mocap_quat[4 * bodyMocapId], quat);
  }
}

inline mjtNum* QueryBodyMocapQuat(const mjModel* model, const mjData* data, const char* body_name) {
  if (data) {
    int bodyMocapId = QueryBodyMocapId(model, body_name);
    return &data->mocap_quat[4 * bodyMocapId];
  }
  return nullptr;
}

inline mjtNum QueryBodyMass(const mjModel* model, int body_id) {
  if (model) {
    return (body_id > -1) ? model->body_mass[body_id] : 0;
  }
  return 0;
}

inline mjtNum QueryBodyMass(const mjModel* model, const char* body_name) {
  if (model) {
    int bodyId = QueryBodyId(model, body_name);
    return (bodyId > -1) ? model->body_mass[bodyId] : 0;
  }
  return 0;
}

// Geom
inline int QueryGeomId(const mjModel* model, const char* geom_name) {
  return model ? mj_name2id(model, mjOBJ_GEOM, geom_name) : -1;
}

inline mjtNum* QueryGeomPos(const mjModel* model, const mjData* data, const char* geom_name) {
  if (data) {
    const int geom_id = QueryGeomId(model, geom_name);
    return (geom_id > -1) ? &data->geom_xpos[3 * geom_id] : nullptr;
  }
  return nullptr;
}

inline mjtNum* QueryGeomQuat(const mjModel* model, const mjData* data, const char* geom_name) {
  if (data) {
    const int geom_id = QueryGeomId(model, geom_name);
    if (geom_id > -1) {
      static mjtNum quat[4];
      mju_mat2Quat(quat, &data->geom_xmat[9 * geom_id]);
      return &quat[0];
    }
  }
  return nullptr;
}

inline std::vector<double> QueryGeomSize(const mjModel* model, const char* geom_name) {
  std::vector<double> size(3, 0.0);
  int geom_id = QueryGeomId(model, geom_name);
  if (geom_id > -1) {
    mju_copy3(size.data(), &model->geom_size[3 * geom_id]);
  }
  return size;
}

inline double QueryGeomSizeMax(const mjModel* model, const char* geom_name) {
  const auto size = QueryGeomSize(model, geom_name);
  return std::max({size[0], size[1], size[2]});
}

inline double QueryGeomSizeMax(const mjModel* model, const std::vector<const char*>& geom_name_list) {
  double max = 0.;
  for (const auto& geom_name : geom_name_list) {
    const auto size = QueryGeomSize(model, geom_name);
    max = std::max(max, std::max({size[0], size[1], size[2]}));
  }
  return max;
}

inline void SetGeomColor(const mjvScene* scene, const mjModel* model, uint geom_id, const float* rgba) {
  if (scene && (geom_id < model->ngeom)) {
    memcpy(scene->geoms[geom_id].rgba, rgba, sizeof(float) * 4);
  }
}

// set mjData state
void SetState(const mjModel* model, mjData* data, const double* state);

// get mjData state
void GetState(const mjModel* model, const mjData* data, double* state);

// get numerical data from a custom element in mjModel with the given name
double* GetCustomNumericData(const mjModel* m, std::string_view name);

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
int ReinterpretAsInt(double value);

// reinterpret int64_t as double
double ReinterpretAsDouble(int64_t value);

// returns a map from custom field name to the list of valid values for that
// field
absl::flat_hash_map<std::string, std::vector<std::string>> ResidualSelectionLists(const mjModel* m);

// get the string selected in a drop down with the given name, given the value
// in the residual parameters vector
std::string ResidualSelection(const mjModel* m, std::string_view name, double residual_parameter);
// returns a value for residual parameters that fits the given text value
// in the given list
double ResidualParameterFromSelection(const mjModel* m, std::string_view name, std::string_view value);

// returns a default value to put in residual parameters, given the index of a
// custom numeric attribute in the model
double DefaultResidualSelection(const mjModel* m, int numeric_index);

// Clamp x between bounds, e.g., bounds[0] <= x[i] <= bounds[1]
void Clamp(double* x, const double* bounds, int n);

// get sensor data using string
double* SensorByName(const mjModel* m, const mjData* d, const std::string& name);

double DefaultParameterValue(const mjModel* model, std::string_view name);

int ParameterIndex(const mjModel* model, std::string_view name);

int CostTermByName(const mjModel* m, const std::string& name);

// return total size of sensors of type user
int ResidualSize(const mjModel* model);

// sanity check that residual size equals total user-sensor dimension
void CheckSensorDim(const mjModel* model, int residual_size);

// get traces from sensors
void GetTraces(double* traces, const mjModel* m, const mjData* d, int num_trace);

// get keyframe `qpos` data using string
double* KeyQPosByName(const mjModel* m, const mjData* d, const std::string& name);

// fills t with N numbers, starting from t0 and incrementing by t_step
void LinearRange(double* t, double t_step, double t0, int N);

// find interval in monotonic sequence containing value
template <typename T>
void FindInterval(int* bounds, const std::vector<T>& sequence, double value, int length) {
  // get bounds
  auto it = std::upper_bound(sequence.begin(), sequence.begin() + length, value);
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
void ZeroInterpolation(double* output, double x, const std::vector<double>& xs, const double* ys, int dim,
                       int length, const std::vector<int>& indices = {});

// linear interpolation
void LinearInterpolation(double* output, double x, const std::vector<double>& xs, const double* ys, int dim,
                         int length, const std::vector<int>& indices = {});

// coefficients for cubic interpolation
void CubicCoefficients(double* coefficients, double x, const std::vector<double>& xs, int T);

// finite-difference vector
double FiniteDifferenceSlope(double x, const std::vector<double>& xs, const double* ys, int dim, int length,
                             int i);

// cubic polynomial interpolation
void CubicInterpolation(double* output, double x, const std::vector<double>& xs, const double* ys, int dim,
                        int length, const std::vector<int>& indices = {});

// returns the path to the directory containing the current executable
std::string GetExecutableDir();

// returns path to a model XML file given path relative to models dir
std::string GetModelPath(std::string_view path);

// dx = (x2 - x1) / h
void Diff(mjtNum* dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n);

// finite-difference two state vectors ds = (s2 - s1) / h
void StateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2, mjtNum h);

// return global height of nearest geom in geomgroup under given position
mjtNum Ground(const mjModel* model, const mjData* data, const mjtNum pos[3],
              const mjtByte* geomgroup = nullptr);

// set x to be the point on the segment [p0 p1] that is nearest to x
void ProjectToSegment(double x[3], const double p0[3], const double p1[3]);

// find frame that best matches 4 feet, z points to body
void FootFrame(double feet_pos[3], double feet_mat[9], double feet_quat[4], const double body[3],
               const double foot0[3], const double foot1[3], const double foot2[3], const double foot3[3]);

// default cost colors
extern const float CostColors[20][3];
constexpr int kNCostColors = sizeof(CostColors) / (sizeof(float) * 3);

// plots - vertical line
void PlotVertical(mjvFigure* fig, double time, double min_value, double max_value, int N, int index);

// plots - update data
void PlotUpdateData(mjvFigure* fig, double* bounds, double x, double y, int length, int index, int x_update,
                    int y_update, double x_bound_lower);

// plots - reset
void PlotResetData(mjvFigure* fig, int length, int index);

// plots - horizontal line
void PlotHorizontal(mjvFigure* fig, const double* xs, double y, int length, int index);

// plots - set data
void PlotData(mjvFigure* fig, double* bounds, const double* xs, const double* ys, int dim, int dim_limit,
              int length, int start_index, double x_bound_lower);

// add geom to scene
void AddGeom(mjvScene* scene, mjtGeom type, const mjtNum size[3], const mjtNum pos[3], const mjtNum mat[9],
             const float rgba[4]);

// add connector geom to scene
void AddConnector(mjvScene* scene, mjtGeom type, mjtNum width, const mjtNum from[3], const mjtNum to[3],
                  const float rgba[4]);

// number of available hardware threads
int NumAvailableHardwareThreads();

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
inline const T* DataAt(const std::vector<T>& vec, typename std::vector<T>::size_type elem) {
  return DataAt(const_cast<std::vector<T>&>(vec), elem);
}

using UniqueMjData = std::unique_ptr<mjData, void (*)(mjData*)>;

inline UniqueMjData MakeUniqueMjData(mjData* d) { return UniqueMjData(d, mj_deleteData); }

using UniqueMjModel = std::unique_ptr<mjModel, void (*)(mjModel*)>;

inline UniqueMjModel MakeUniqueMjModel(mjModel* d) { return UniqueMjModel(d, mj_deleteModel); }

// returns point in 2D convex hull that is nearest to query
void NearestInHull(mjtNum res[2], const mjtNum query[2], const mjtNum* points, const int* hull, int num_hull);

// find the convex hull of a set of 2D points
int Hull2D(int* hull, int num_points, const mjtNum* points);

// TODO(etom): move findiff-related functions to a different library.

// finite-difference gradient
class FiniteDifferenceGradient {
public:
  // constructor
  explicit FiniteDifferenceGradient(int dim);

  // resize memory
  void Resize(int dim);

  // compute gradient
  void Compute(std::function<double(const double* x)> func, const double* input, int dim);

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
  void Compute(std::function<void(double* output, const double* input)> func, const double* input,
               int num_output, int num_input);

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
  void Compute(std::function<double(const double* x)> func, const double* input, int dim);

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
void SetBlockInMatrix(double* mat, const double* block, double scale, int rm, int cm, int rb, int cb, int ri,
                      int ci);

// set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void AddBlockInMatrix(double* mat, const double* block, double scale, int rm, int cm, int rb, int cb, int ri,
                      int ci);

// get block (size: rb x cb) from mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void BlockFromMatrix(double* block, const double* mat, int rb, int cb, int rm, int cm, int ri, int ci);

// differentiate mju_subQuat wrt qa, qb
void DifferentiateSubQuat(double jaca[9], double jacb[9], const double qa[4], const double qb[4]);

// differentiate velocity by finite-differencing two positions wrt to qpos1,
// qpos2
void DifferentiateDifferentiatePos(double* jac1, double* jac2, const mjModel* model, double dt,
                                   const double* qpos1, const double* qpos2);

// compute number of nonzeros in band matrix
int BandMatrixNonZeros(int ntotal, int nband);

// TODO(etom): rename (SecondsSince?)
double GetDuration(std::chrono::steady_clock::time_point time);

// copy symmetric band matrix block by block
void SymmetricBandMatrixCopy(double* res, const double* mat, int dblock, int nblock, int ntotal,
                             int num_blocks, int res_start_row, int res_start_col, int mat_start_row,
                             int mat_start_col, double* scratch);

// zero block (size: rb x cb) in mat (size: rm x cm) given mat upper row
// and left column indices (ri, ci)
void ZeroBlockInMatrix(double* mat, int rm, int cm, int rb, int cb, int ri, int ci);

// square dense to block band matrix
void DenseToBlockBand(double* res, int dim, int dblock, int nblock);

// infinity norm
template <typename T>
T InfinityNorm(T* x, int n) {
  return std::abs(*std::max_element(x, x + n, [](T a, T b) -> bool { return (std::abs(a) < std::abs(b)); }));
}

// trace of square matrix
double Trace(const double* mat, int n);

// determinant of 3x3 matrix
double Determinant3(const double* mat);

// inverse of 3x3 matrix
void Inverse3(double* res, const double* mat);

// condition matrix: res = mat11 - mat10 * mat00 \ mat10^T; return rank of mat00
// TODO(taylor): thread
void ConditionMatrix(double* res, const double* mat, double* mat00, double* mat10, double* mat11,
                     double* tmp0, double* tmp1, int n, int n0, int n1, double* bandfactor = NULL,
                     int nband = 0);

// principal eigenvector of 4x4 matrix
// QUEST algorithm from "Three-Axis Attitude Determination from Vector
// Observations"
void PrincipalEigenVector4(double* res, const double* mat, double eigenvalue_init = 12.0);

// set scaled symmetric block matrix in band matrix
void SetBlockInBand(double* band, const double* block, double scale, int ntotal, int nband, int nblock,
                    int shift, int row_skip = 0, bool add = true);
} // namespace mjpc

#endif  // MJPC_UTILITIES_H_
