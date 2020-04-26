/* 
 * strongscaling_ridgeregression_hogwild.cpp
 *
 * Benchmark the 
 *
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

#include "readCSV.h"

typedef Eigen::VectorXd Evec;
typedef Eigen::MatrixXd Emat;

using namespace Eigen;

/*
 *
 */
double
obj(

int 
main(int argc, char **argv)
{
  int P = omp_get_max_threads();


  return 0;
}
