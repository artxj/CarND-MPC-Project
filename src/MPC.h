#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC {
  unsigned int freeze_latency_count; // how many timesteps we need to freeze due to latency

  double ref_v; // reference velocity

  // notice: we need to store only one pair [delta, a]
  // because when we sleep for 100ms, during all that time only one actuation takes place
  Eigen::VectorXd stored_actuations;

 public:
  /* MPC with latency provided
  @param latency Latency before control takes in place
  @param ref_velocity Velocity the vehicle wants to reach
  */
  MPC(const double latency, const double ref_velocity);

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  // MPC predicted trajectories
  vector<double> mpc_x_vals;
  vector<double> mpc_y_vals;
};

#endif /* MPC_H */
