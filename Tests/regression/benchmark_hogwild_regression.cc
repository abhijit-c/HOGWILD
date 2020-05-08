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
#define ETA 0.005
#define TOL 0.00001
#define MAX_ITERATIONS 10000
#define TRIALS_PER_CORE 10

int 
main(int argc, char **argv)
{
  Eigen::initParallel();
  unsigned P = omp_get_max_threads();
  double timings[P];

  // Read MSD dataset into memory and format matrices.
  Eigen::MatrixXd Data;
  readCSV<double>("simplemat.csv", Data);

  unsigned num_data = Data.rows(); unsigned num_features = Data.cols()-1;

  Eigen::MatrixXd A = Data.topRightCorner(num_data, num_features);
  Eigen::VectorXd b = Data.col(0);

  srand (1);
  Eigen::VectorXd x = Eigen::VectorXd::Ones(b.size());
  double t_start, t_end;
  t_start = omp_get_wtime();
  
  unsigned it = 1; double learning_rate = ETA;
  while ( 2.0*(A*x-b).norm() >= TOL && it <= MAX_ITERATIONS )
  {
    unsigned i = rand() % num_data;
    x -= learning_rate*( 2 * A.row(i).transpose()*( A.row(i)*x - b(i) ) );
    //x -= learning_rate*( 2*A.transpose()*(A*x-b) );
    ++it;
  }

  t_end = omp_get_wtime();
  printf("Time elapsed: %.4f\n", t_end-t_start);
  std::cout << "Residual: " << (A*x-b).norm() << std::endl;
  return 0;
}
