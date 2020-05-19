/* 
 * benchmark_hogwild_regression.cpp
 * author: Abhijit Chowdhary (achowdh2@ncsu.edu)
 *
 * Benchmark HOGWILD! as applied to regression on to a simple random normal 50
 * x 50 matrix A and random normal vector b:
 *
 *  minimize (1/2)||Ax-b||_2^2
 *
 * Outputs to results.txt and stdout time taken to reach desired tolerance for
 * each core count possible in system.
 *  ---------------------------------------------------------------------------
 * This Source Code Form is subject to the terms of the Mozilla Public License 
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can 
 * obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <omp.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "compute_variance.h"

#define N 1024
#define ETA 0.00001
#define NUM_EPOCHS 5
#define TRIALS_PER_BAND 100
#define MAX_BAND 1

int 
main(int argc, char **argv)
{
  // Initialize random state, and parallel parameters.
  unsigned P = omp_get_max_threads();
  Eigen::initParallel();
  omp_set_dynamic(0);
  std::mt19937 gen(0);

  // Construct sampling w/ replacement vector.
  std::uniform_int_distribution<std::mt19937::result_type> distN(0,N-1);
  unsigned *rand_selection = new unsigned[N*NUM_EPOCHS];
  for (unsigned k = 0; k < N*NUM_EPOCHS; k++)
  {
    rand_selection[k] = distN(gen);
  }
  
  std::vector<double> computed_variances;
  Eigen::MatrixXd X = Eigen::MatrixXd::Constant(N, 1, 1.0);
  for (unsigned band = 0; band < 1; band++)
  {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, N);
    A.diagonal(0) = Eigen::MatrixXd::Random(N,1);
    for (int k = 1; k <= band; k++)
    {
      A.diagonal(k) = Eigen::MatrixXd::Random(N-k, 1);
      A.diagonal(-k) = Eigen::MatrixXd::Random(N-k, 1);
    }

    Eigen::VectorXd b = A*X;

    std::vector<double> computed_norms;
    std::array<std::atomic<double>, N> x;
    for (unsigned trial = 0; trial < TRIALS_PER_BAND; ++trial)
    { // Begin SGD trial
      for (unsigned k = 0; k < N; k++) { x[k] = 0.0; }

      #pragma omp parallel for
      for (unsigned k = 0; k < N*NUM_EPOCHS; k++)
      { // Begin parallel SGD iterations
        unsigned id = rand_selection[k];
        double dg = 0;
        for (unsigned i = 0; i < N; i++) { dg += A(id, i)*x[i].load(); }
        dg -= b(id);
        for (unsigned i = 0; i < N; i++)
        {
          double dgi = x[i].load() - ETA*( A(id,i)*dg );
          x[i].exchange( dgi );
        }
      } // End parallel SGD iterations

      Eigen::MatrixXd xx(N,1);
      for (int k = 0; k < N; k++) { xx(k) = x[k].load(); }
      computed_norms.push_back( 0.5*(A*xx-b).squaredNorm() );
    } // End SGD trial

    computed_variances.push_back( compute_variance(computed_norms) );
  }
  for (unsigned k = 0; k < computed_variances.size(); k++)
  {
    printf("Var(B=k) = %.9f\n", computed_variances[k]);
  }
  delete []rand_selection;
  return 0;
}
