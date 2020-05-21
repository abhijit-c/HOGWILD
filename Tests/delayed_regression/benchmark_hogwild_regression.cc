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

#define ETA 0.01
#define NUM_EPOCHS 50
#define TRIALS_PER_CORE 3

int 
main(int argc, char **argv)
{
  Eigen::initParallel();
  omp_set_dynamic(0);
  auto rng = std::default_random_engine {}; rng.seed(0);

  unsigned P = omp_get_max_threads();
  double timings[P];
  for (unsigned k = 0; k < P; k++) { timings[k] = 0; }

  // Read MSD dataset into memory and format matrices.
  Eigen::MatrixXd Data;
  readCSV<double>("simplemat.csv", Data);
  unsigned const num_data = 50; unsigned const num_features = 50;

  Eigen::MatrixXd A = Data.topRightCorner(num_data, num_features);
  Eigen::VectorXd b = Data.col(0);

  std::array<unsigned, num_data> ordering;
  std::iota(ordering.begin(), ordering.end(), 0);

  std::array<std::atomic<double>, num_features> x;
  for (unsigned p = 0; p < P; ++p)
  {
    omp_set_num_threads(p+1);
    for (unsigned trial = 0; trial < TRIALS_PER_CORE; ++trial)
    {
      for (unsigned k = 0; k < num_features; k++) { x[k] = 1; }

      double t_start, t_end;
      t_start = omp_get_wtime();
      for (unsigned epoch = 0; epoch < NUM_EPOCHS; epoch++)
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
            double dgi = x[i].load() - ETA*( A(id,i)*dg );
            x[i].exchange( dgi );
          }
          double tt = omp_get_wtime();
          double ttt = omp_get_wtime();
          while (ttt < tt+.05) { ttt = omp_get_wtime(); }
        }
        printf("%d\n", epoch); 
      }
      t_end = omp_get_wtime();
      timings[p] += t_end-t_start;
    }
    timings[p] /= TRIALS_PER_CORE;
  }
  Eigen::MatrixXd xx(num_features,1);
  for (int k = 0; k < num_features; k++) { xx(k) = x[k].load(); }
  std::cout << 0.5*(A*xx-b).squaredNorm() << std::endl;

  for (unsigned k = 0; k < P; k++)
  {
    printf("T(p=%d) = %.5f\n", k+1, timings[k]);
  }
  return 0;
}
