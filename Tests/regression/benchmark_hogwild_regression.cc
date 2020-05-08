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
#include <random>

#include <Eigen/Dense>

#include "readCSV.h"

#define ETA 0.01
#define TRIALS_PER_CORE 10

int 
main(int argc, char **argv)
{
  Eigen::initParallel();
  omp_set_dynamic(0);
  auto rng = std::default_random_engine {1};

  unsigned P = omp_get_max_threads();
  double timings[P];

  // Read MSD dataset into memory and format matrices.
  Eigen::MatrixXd Data;
  readCSV<double>("simplemat.csv", Data);
  unsigned const num_data = 50; unsigned const num_features = 50;

  Eigen::MatrixXd A = Data.topRightCorner(num_data, num_features);
  Eigen::VectorXd b = Data.col(0);

  std::array<std::atomic<double>, num_features> x;
  for (unsigned k = 0; k < num_features; k++) { x[k] = 1; }

  std::array<unsigned, num_data> ordering;
  std::iota(ordering.begin(), ordering.end(), 0);

  double t_start, t_end;
  t_start = omp_get_wtime();
  double learning_rate = ETA;
  for (unsigned epoch = 0; epoch < 20; epoch++)
  {
    std::shuffle(ordering.begin(), ordering.end(), rng);
    #pragma omp parallel for
    for (unsigned k = 0; k < num_data; k++)
    {
      unsigned id = ordering[k];
      double dg = 0;
      for (unsigned i = 0; i < num_features; i++) { dg += A(id, i)*x[i].load(); }
      dg -= b(id);
      for (unsigned i = 0; i < num_features; i++)
      {
        double dgi = x[i].load() - learning_rate*( A(id,i)*dg );
        x[i].exchange( dgi );
      }
    }
  }
  t_end = omp_get_wtime();

  Eigen::VectorXd x_comp (num_features);
  for (unsigned k = 0; k < num_features; k++) { x_comp(k) = x[k].load(); }

  printf("Time elapsed: %.4f\n", t_end-t_start);
  printf("Residual: %.4f\n", (A*x_comp-b).norm());
  return 0;
}
