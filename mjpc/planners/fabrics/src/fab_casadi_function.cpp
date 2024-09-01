#include "mjpc/planners/fabrics/include/fab_casadi_function.h"

#define USE_PYTHON_FUNC (0)

#if USE_PYTHON_FUNC
#include "mjpc/planners/fabrics/include/casadi_gen/fab_task_panda_dynamic_python_func.h"
#else
#include "mjpc/planners/fabrics/include/casadi_gen/fab_task_panda_dynamic_func.h"
#endif
#include "mjpc/planners/fabrics/include/fab_core_util.h"

#define SET_INPUT_FROM_ARG1(idx, key)                   \
  {                                                     \
    const auto value = arguments_.at(key);              \
    input[idx] = (double[]){(double)value(0).scalar()}; \
  }

#define CASX_TO_ARRAY3(input) \
  (double[]) { (double)input(0).scalar(), (double)input(1).scalar(), (double)input(2).scalar() }

#define SET_INPUT_FROM_ARG3(idx, key)      \
  {                                        \
    const auto value = arguments_.at(key); \
    input[idx] = CASX_TO_ARRAY3(value);    \
  }

void FabCasadiFunction::setup_pregen_functions() {
  pregen_function_list_[fab_core::task_function_name("PickAndPlace", false, true)] = [this]() {
    unsigned int nb_inputs = 16;
    unsigned int nb_outputs = 1;
    const double** input = new const double*[nb_inputs];

#if USE_PYTHON_FUNC
    const auto q = arguments_.at("q");
    input[0] =
        (double[]){(double)q(0).scalar(), (double)q(1).scalar(), (double)q(2).scalar(), (double)q(3).scalar(),
                   (double)q(4).scalar(), (double)q(5).scalar(), (double)q(6).scalar()};
    const auto qdot = arguments_.at("qdot");
    input[1] = (double[]){(double)qdot(0).scalar(), (double)qdot(1).scalar(), (double)qdot(2).scalar(),
                          (double)qdot(3).scalar(), (double)qdot(4).scalar(), (double)qdot(5).scalar(),
                          (double)qdot(6).scalar()};
    SET_INPUT_FROM_ARG3(2, "x_obst_0");
    SET_INPUT_FROM_ARG1(3, "radius_obst_0");
    SET_INPUT_FROM_ARG1(4, "radius_body_panda_link9");
    SET_INPUT_FROM_ARG3(5, "x_obst_1");
    SET_INPUT_FROM_ARG1(6, "radius_obst_1");
    SET_INPUT_FROM_ARG3(7, "x_obst_2");
    SET_INPUT_FROM_ARG1(8, "radius_obst_2");
    SET_INPUT_FROM_ARG3(9, "constraint_0");
    SET_INPUT_FROM_ARG1(10, "radius_body_panda_link8");
    SET_INPUT_FROM_ARG1(11, "radius_body_panda_link4");
    SET_INPUT_FROM_ARG3(12, "x_ref_goal_0_leaf");
    SET_INPUT_FROM_ARG3(13, "xdot_ref_goal_0_leaf");
    SET_INPUT_FROM_ARG3(14, "xddot_ref_goal_0_leaf");
    SET_INPUT_FROM_ARG1(15, "weight_goal_0");
#else
    SET_INPUT_FROM_ARG3(0, "constraint_0");
    const auto q = arguments_.at("q");
    input[1] =
        (double[]){(double)q(0).scalar(), (double)q(1).scalar(), (double)q(2).scalar(), (double)q(3).scalar(),
                   (double)q(4).scalar(), (double)q(5).scalar(), (double)q(6).scalar()};
    const auto qdot = arguments_.at("qdot");
    input[2] = (double[]){(double)qdot(0).scalar(), (double)qdot(1).scalar(), (double)qdot(2).scalar(),
                          (double)qdot(3).scalar(), (double)qdot(4).scalar(), (double)qdot(5).scalar(),
                          (double)qdot(6).scalar()};
    SET_INPUT_FROM_ARG1(3, "radius_body_panda_link4");
    SET_INPUT_FROM_ARG1(4, "radius_body_panda_link8");
    SET_INPUT_FROM_ARG1(5, "radius_body_panda_link9");
    SET_INPUT_FROM_ARG1(6, "radius_obst_0");
    SET_INPUT_FROM_ARG1(7, "radius_obst_1");
    SET_INPUT_FROM_ARG1(8, "radius_obst_2");
    SET_INPUT_FROM_ARG1(9, "weight_goal_0");
    SET_INPUT_FROM_ARG3(10, "x_obst_0");
    SET_INPUT_FROM_ARG3(11, "x_obst_1");
    SET_INPUT_FROM_ARG3(12, "x_obst_2");

    SET_INPUT_FROM_ARG3(13, "x_ref_goal_0_leaf");
    SET_INPUT_FROM_ARG3(14, "xddot_ref_goal_0_leaf");
    SET_INPUT_FROM_ARG3(15, "xdot_ref_goal_0_leaf");
#endif

    double** output = new double*[nb_outputs];
    output[0] = new double[9];

    long long int* setting_0 = new long long int[2];
    double* setting_1 = new double[2];
    int setting_2 = 0;

#if USE_PYTHON_FUNC
    fab_task_panda_python::casadi_f0(input, output, setting_0, setting_1, setting_2);
#else
    fab_task_panda_dynamic::casadi_f0(input, output, setting_0, setting_1, setting_2);
#endif

    std::vector<double> outputs(9, 0);
    memcpy(outputs.data(), &output[0][0], 9 * sizeof(double));
    return outputs;
  };
}

CaSX FabCasadiFunction::call_pregen_function() const {
  return pregen_function_list_.contains(name_) ? pregen_function_list_.at(name_)() : CaSX::zeros();
}