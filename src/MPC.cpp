#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// Set the timestep length and duration
const size_t N = 20;
const double dt = 0.05;

// start indices for the state and actuators parameters
const size_t x_start = 0;
const size_t y_start = x_start + N;
const size_t ksi_start = y_start + N;
const size_t v_start = ksi_start + N;
const size_t cte_start = v_start + N;
const size_t eksi_start = cte_start + N;
const size_t delta_start = eksi_start + N;
const size_t a_start = delta_start + N - 1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

class FG_eval {
  double ref_v; // reference velocity

 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;

  FG_eval(Eigen::VectorXd coeffs, double ref_v) {
    this->coeffs = coeffs;
    this->ref_v = ref_v;
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // Implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // calculating the cost
    fg[0] = 0;

    // minimize cost of errors (cte, direction, velocity)
    for (unsigned int i = 0; i < N; ++i) {
      fg[0] += CppAD::pow(vars[cte_start + i], 2);
      fg[0] += CppAD::pow(vars[eksi_start + i], 2);
      fg[0] += CppAD::pow(vars[v_start + i] - ref_v, 2);
    }

    // minimize actuations values to prevent sharp actions
    for (unsigned int i = 0; i < N - 1; i++) {
      fg[0] += CppAD::pow(vars[delta_start + i], 2);
      fg[0] += CppAD::pow(vars[a_start + i], 2);
    }

    // minimize gap between sequential actuations
    const AD<double> delta_diff_coeff = 1000; // to compensate oscillation
    for (unsigned int i = 0; i < N - 2; i++) {
      fg[0] += delta_diff_coeff * CppAD::pow(vars[delta_start + i + 1] - vars[delta_start + i], 2);
      fg[0] += CppAD::pow(vars[a_start + i + 1] - vars[a_start + i], 2);
    }

    // initial state
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + ksi_start] = vars[ksi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + eksi_start] = vars[eksi_start];

    // constraints
    for (unsigned int i = 1; i < N; ++i) {
      // state at time t
      const AD<double> x0 = vars[x_start + i - 1];
      const AD<double> y0 = vars[y_start + i - 1];
      const AD<double> ksi0 = vars[ksi_start + i - 1];
      const AD<double> v0 = vars[v_start + i - 1];
      const AD<double> cte0 = vars[cte_start + i - 1];
      const AD<double> eksi0 = vars[eksi_start + i - 1];

      // state at time t+1
      const AD<double> x1 = vars[x_start + i];
      const AD<double> y1 = vars[y_start + i];
      const AD<double> ksi1 = vars[ksi_start + i];
      const AD<double> v1 = vars[v_start + i];
      const AD<double> cte1 = vars[cte_start + i];
      const AD<double> eksi1 = vars[eksi_start + i];

      // actuations at time t
      const AD<double> delta0 = vars[delta_start + i - 1];
      const AD<double> a0 = vars[a_start + i - 1];

      const AD<double> x0sq = x0 * x0;
      const AD<double> x0th = x0 * x0sq;
      const AD<double> f0 = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0sq + coeffs[3] * x0th;
      const AD<double> ksides0 = CppAD::atan(coeffs[1] + 2 * coeffs[2] * x0 + 3 * coeffs[3] * x0sq); // derivative

      // x_[t] = x[t-1] + v[t-1] * cos(ksi[t-1]) * dt
      // y_[t] = y[t-1] + v[t-1] * sin(ksi[t-1]) * dt
      // ksi_[t] = ksi[t-1] + v[t-1] / Lf * delta[t-1] * dt // changed sign due to sim turn sign
      // v_[t] = v[t-1] + a[t-1] * dt
      // cte[t] = f(x[t-1]) - y[t-1] + v[t-1] * sin(eksi[t-1]) * dt
      // eksi[t] = ksi[t] - ksides[t-1] + v[t-1] * delta[t-1] / Lf * dt
      fg[1 + x_start + i] = x1 - (x0 + v0 * CppAD::cos(ksi0) * dt);
      fg[1 + y_start + i] = y1 - (y0 + v0 * CppAD::sin(ksi0) * dt);
      fg[1 + ksi_start + i] = ksi1 - (ksi0 + v0 * delta0 * dt / Lf);
      fg[1 + v_start + i] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + i] = cte1 - (f0 - y0 + v0 * CppAD::sin(eksi0) * dt);
      fg[1 + eksi_start + i] = eksi1 - (ksi0 - ksides0 + v0 * delta0 * dt / Lf);
    }
  }

};

//
// MPC class definition implementation.
//
MPC::MPC(const double latency, const double ref_velocity) {
  freeze_latency_count = unsigned(ceilf(latency / dt));
  stored_actuations = Eigen::VectorXd(2);
  stored_actuations.setZero();
  mpc_x_vals = vector<double>(N);
  mpc_y_vals = vector<double>(N);
  ref_v = ref_velocity;
}

MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  // 6 elements in state [x, y, ksi, v, cte, eksi]
  // 2 actuators [delta, a]
  // N timesteps
  const size_t n_vars = 6 * N + 2 * (N - 1);
  // Set the number of constraints
  const size_t n_constraints = 6 * N;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // initial state
  const double x = state[0];
  const double y = state[1];
  const double ksi = state[2];
  const double v = state[3];
  const double cte = state[4];
  const double eksi = state[5];
  vars[x_start] = x;
  vars[y_start] = y;
  vars[ksi_start] = ksi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[eksi_start] = eksi;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // Set lower and upper limits for variables.
  // state variables - can be anything
  const double dmax = std::numeric_limits<double>::max() - 1;
  for (unsigned int i = 0; i < delta_start; ++i) {
    vars_lowerbound[i] = -dmax;
    vars_upperbound[i] = dmax;
  }

  // delta for latency period is equal to already stored actuations
  const double stored_delta = stored_actuations[0];
  for (unsigned int i = delta_start; i < delta_start + freeze_latency_count; ++i) {
    vars_lowerbound[i] = vars_upperbound[i] = stored_delta;
  }
  // future delta
  for (unsigned int i = delta_start + freeze_latency_count; i < a_start; ++i) {
    // [-25, 25] deg transformed to rads
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // a for latency period
  const double stored_a = stored_actuations[1];
  for (unsigned int i = a_start; i < a_start + freeze_latency_count; ++i) {
    vars_lowerbound[i] = vars_upperbound[i] = stored_a;
  }
  // future a
  for (unsigned int i = a_start + freeze_latency_count; i < n_vars; ++i) {
    vars_lowerbound[i] = -1;
    vars_upperbound[i] = 1;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  // initial state
  constraints_lowerbound[x_start] = constraints_upperbound[x_start] = x;
  constraints_lowerbound[y_start] = constraints_upperbound[y_start] = y;
  constraints_lowerbound[ksi_start] = constraints_upperbound[ksi_start] = ksi;
  constraints_lowerbound[v_start] = constraints_upperbound[v_start] = v;
  constraints_lowerbound[cte_start] = constraints_upperbound[cte_start] = cte;
  constraints_lowerbound[eksi_start] = constraints_upperbound[eksi_start] = eksi;


  // object that computes objective and constraints
  FG_eval fg_eval(coeffs, ref_v);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.

  // updating stored actuations
  // first one will be removed at next cycle, other steps back
  // and new one will be obtained from the current solution

  // apply actuations that come in place after given delay
  const double solution_delta = solution.x[delta_start + freeze_latency_count];
  const double solution_a = solution.x[a_start + freeze_latency_count];

  stored_actuations[0] = solution_delta;
  stored_actuations[1] = solution_a;

  // MPC predicted trajectories
  for (unsigned int i = 0; i < N; ++i) {
    mpc_x_vals[i] = solution.x[x_start + i];
    mpc_y_vals[i] = solution.x[y_start + i];
  }

  return { solution_delta, solution_a };
}
