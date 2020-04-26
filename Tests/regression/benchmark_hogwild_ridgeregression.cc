/* 
 * benchmark_hogwild_ridgeregression.cpp
 * author: Abhijit Chowdhary (achowdh2@ncsu.edu)
 *
 * Benchmark HOGWILD! as applied to ridge regression of given dataset:
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
#include <stdio.h>
#include <omp.h>

#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "ridgeregression_functions.h"
#include "readCSV.h"

int 
main(int argc, char **argv)
{
  int P = omp_get_max_threads();

  return 0;
}
