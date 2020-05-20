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

#include <Eigen/Dense>

#include "readCSV.h"

#define ETA 0.00001
#define NUM_EPOCHS 50

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
  std::cout << Data.rows() << std::endl;
  std::cout << Data.cols() << std::endl;
  unsigned const num_data = 400; unsigned const num_features = 160000;

  Eigen::MatrixXd A = Data.topRightCorner(num_data, num_features);
  Eigen::VectorXd b = Data.col(0);

  std::array<unsigned, num_data> ordering;
  std::iota(ordering.begin(), ordering.end(), 0);

  //std::array<std::atomic<double>, num_features> x;
  std::atomic<double> *x = new std::atomic<double>[num_features];
  Eigen::MatrixXd X(num_features,1);
  for (unsigned k = 0; k < num_features; k++) { x[k] = 1; X(k) = 1; }
  printf("%f\n", 0.5*(A*X-b).squaredNorm());

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
    }
    for (unsigned k = 0; k < num_features; k++) { X(k) = x[k].load(); }
    double epsilon = 0.5*(A*X-b).squaredNorm();
    printf("%f\n", epsilon);
    if (epsilon < 1e-5)
    {
      break;
    }
  }

  return 0;
}
