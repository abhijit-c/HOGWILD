/* 
 * benchmark_hogwild_regression.cpp
 * author: Abhijit Chowdhary (achowdh2@ncsu.edu)
 *
 * Benchmark HOGWILD! as applied to regression on to a simple random normal 50
 * x 50 matrix A and random normal vector b:
 *
 *  minimize ||Ax-b||_2^2
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

#include <Eigen/Dense>

#include "readCSV.h"

#define ETA 0.001
#define NUM_EPOCHS 50
#define TRIALS_PER_CORE 5

int 
main(int argc, char **argv)
{
  // Initialize random state, and parallel parameters.
  unsigned P = omp_get_max_threads();
  Eigen::initParallel();
  omp_set_dynamic(0);
  std::mt19937 gen(0);

  // Read MSD dataset into memory and format matrices.
  unsigned const num_data = 160000; unsigned const num_features = 400;
  Eigen::MatrixXd Data;
  readCSV<double>("../simplemat.csv", Data);
  Eigen::MatrixXd A = Data.topRightCorner(num_data, num_features);
  Eigen::VectorXd b = Data.col(0);

  // Construct sampling w/ replacement vector.
  std::uniform_int_distribution<std::mt19937::result_type> distN(0,num_data-1);
  //std::array<unsigned, num_data*NUM_EPOCHS> rand_selection;
  unsigned *rand_selection = new unsigned[num_data*NUM_EPOCHS];
  for (unsigned k = 0; k < num_data*NUM_EPOCHS; k++)
  {
    rand_selection[k] = distN(gen);
  }

  std::array<std::atomic<double>, num_features> x;
  for (unsigned p = 0; p < P; ++p)
  { // Begin trials for processor count p+1
    omp_set_num_threads(p+1);
    double t = 0.0;
    for (unsigned trial = 0; trial < TRIALS_PER_CORE; ++trial)
    { // Begin SGD trial
      for (unsigned k = 0; k < num_features; k++) { x[k] = 1; }

      double t_start, t_end;
      t_start = omp_get_wtime();

      #pragma omp parallel for
      for (unsigned k = 0; k < num_data*NUM_EPOCHS; k++)
      { // Begin parallel SGD iterations
        unsigned id = rand_selection[k];
        double dg = 0;
        for (unsigned i = 0; i < num_features; i++) { dg += A(id, i)*x[i].load(); }
        dg -= b(id);
        for (unsigned i = 0; i < num_features; i++)
        {
          double dgi = x[i].load() - ETA*( A(id,i)*dg );
          x[i].exchange( dgi );
        }
      } // End parallel SGD iterations

      t_end = omp_get_wtime(); t += t_end-t_start;
    } // End SGD trial
    t /= TRIALS_PER_CORE;

    Eigen::MatrixXd xx(num_features,1);
    for (int k = 0; k < num_features; k++) { xx(k) = x[k].load(); }
    double E = 0.5*(A*xx-b).squaredNorm();
    printf("T(p=%d) = %.5f, E(p=%d) = %.5f\n", p+1, t, p+1, E);
  } // End trials for processor count p+1

  delete []rand_selection;
  return 0;
}
