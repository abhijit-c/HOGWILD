/* 
 * benchmark_hogwild_ridgeregression.cpp
 * author: Abhijit Chowdhary (achowdh2@ncsu.edu)
 *
 * Benchmark HOGWILD! as applied to ridge regression on a subset of the Million
 * Song Dataset:
 *
 *  minimize ||Ax-b||_2^2 + \lambda||x||_2^2
 *
 * Outputs to results.txt and stdout time taken to reach desired tolerance for
 * each core count possible in system.
 *  ---------------------------------------------------------------------------
 * This Source Code Form is subject to the terms of the Mozilla Public License 
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can 
 * obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "readCSV.h"

#define TOL 100.0
#define LAMBDA 2.0
#define TRIALS_PER_CORE 10

int 
main(int argc, char **argv)
{
  unsigned P = omp_get_max_threads();
  double timings[P];

  // Read MSD dataset into memory and format matrices.
  Eigen::MatrixXd Data;
  readCSV<double>("YearPredictionMSD.csv", Data);

  unsigned num_data = Data.rows(); unsigned num_features = Data.cols()-1;
  unsigned num_train = int(Data.rows()*0.9); unsigned num_test = Data.rows()-num_train;

  Eigen::MatrixXd A = Data.topRightCorner(num_train, num_features);
  Eigen::MatrixXd T = Data.bottomRightCorner(num_test, num_features);
  Eigen::VectorXd b = Data.col(0);

  for (unsigned p = 1; p <= P; ++p)
  { //Perform HOGWILD based ridge regression w/ p cores.
    omp_set_num_threads(p);

    for (unsigned trial = 0; trial < TRIALS_PER_CORE; ++trial);
    {
      srand (1);
      double t_start, t_end;
      Eigen::VectorXd x(b.size());

      t_start = omp_get_wtime();
      
      unsigned it = 1;
      while ( (T*x-b).squaredNorm() + LAMBDA*x.squaredNorm() >= TOL )
      {
        unsigned i = ran
        #pragma omp parallel for
        for (unsigned k = 0; k < p; i++)
        {
          unsigned i = rand() % num_train;
          x = x - 2* //TODO: Complete update step
        }

      }

      t_end = omp_get_wtime();
      timings[p] += t_end-t_start;
    }
    timings[p] /= TRIALS_PER_CORE;
  }

  return 0;
}
