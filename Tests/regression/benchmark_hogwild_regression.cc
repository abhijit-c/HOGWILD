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
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include <fstream>
#include <vector>

#include <Eigen/Dense>

#include "readCSV.h"

#define LAMBDA 2.0
#define ETA 0.1
#define TOL 100.0
#define MAX_ITERATIONS 1000
#define TRIALS_PER_CORE 10

int 
main(int argc, char **argv)
{
  unsigned P = omp_get_max_threads();
  double timings[P];

  // Read MSD dataset into memory and format matrices.
  Eigen::MatrixXd Data;
  readCSV<double>("simplemat.csv", Data);

  unsigned num_data = Data.rows(); unsigned num_features = Data.cols()-1;

  Eigen::MatrixXd A = Data.topRightCorner(num_data, num_features);
  Eigen::VectorXd b = Data.col(0);
  Eigen::VectorXd x(b.size()); Eigen::MatrixXd X(b.size(), P);

  for (unsigned p = 1; p <= P; ++p)
  { //Perform HOGWILD based ridge regression w/ p cores.
    omp_set_num_threads(p);
    unsigned num_epochs = num_data/p + (num_data % p != 0); //ceil(num_data/p)

    for (unsigned trial = 0; trial < TRIALS_PER_CORE; ++trial);
    { //Average runtime over number of trials
      srand (1);
      double t_start, t_end;
      t_start = omp_get_wtime();
      
      unsigned it = 1; double learning_rate = ETA;
      while ( 2*(A*x-b).norm() >= TOL && it <= MAX_ITERATIONS )
      {
        for (unsigned epoch = 0; epoch < num_epochs; epoch++)
        {
          #pragma omp parallel for
          for (unsigned k = 0; k < p; k++)
          {
            unsigned i = rand() % num_data;
            x = x - learning_rate*( 2*(A.row(i)*x - b) );
          }
          learning_rate = learning_rate / 2;
        }
      }

      t_end = omp_get_wtime();
      timings[p] += t_end-t_start;
    }

    X.col(p-1) = x;
    timings[p] /= TRIALS_PER_CORE;
  }

  for (int p = 0; p < P; p++) { std::cout << timings[p] << ", "; }
  std::cout << std::endl;
  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file("results.txt");
  file << X.format(CSVFormat);
  return 0;
}
