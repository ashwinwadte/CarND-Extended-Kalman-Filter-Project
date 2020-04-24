#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  /**
   * TODO: Calculate the RMSE here.
   */
  
  VectorXd rmse(4);
   rmse << 0, 0, 0, 0;

   // check the validity of the following inputs:
   //  1. the estimation vector size should not be zero
   //  2. the estimation vector size should equal ground truth vector size
   if (estimations.size() != ground_truth.size() || estimations.size() == 0)
   {
      cout << "CalculateRMSE: Invalid estimation or ground_truth data" << endl;
      return rmse;
   }

   // accumulate squared residuals
   for (unsigned int i = 0; i < estimations.size(); ++i)
   {

      VectorXd residual = estimations[i] - ground_truth[i];

      // coefficient-wise multiplication
      residual = residual.array() * residual.array();
      rmse += residual;
   }

   // calculate the mean
   rmse = rmse / estimations.size();

   // calculate the squared root
   rmse = rmse.array().sqrt();

   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
  /**
   * TODO:
   * Calculate a Jacobian here.
   */

   MatrixXd Hj(3, 4);

   // Recover state parameters
   float p_x = x_state(0);
   float p_y = x_state(1);
   float v_x = x_state(2);
   float v_y = x_state(3);

   // Pre-compute a set of terms to avoid repeated calculation
   float c1 = p_x * p_x + p_y * p_y;
   float c2 = sqrt(c1);
   float c3 = (c1 * c2);

   // check division by zero
   if (fabs(c1) < 0.0001)
   {
      cout << "CalculateJacobian: Error - Division by Zero" << endl;
      return Hj;
   }

   // compute the Jacobian matrix
   Hj << (p_x / c2), (p_y / c2), 0, 0,
       -(p_y / c1), (p_x / c1), 0, 0,
       p_y * (v_x * p_y - v_y * p_x) / c3, p_x * (p_x * v_y - p_y * v_x) / c3, p_x / c2, p_y / c2;

   return Hj;
}
