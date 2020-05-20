#ifndef COMPUTE_VARIANCE_H
#define COMPUTE_VARIANCE_H

#include <Eigen/Core>

#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

inline double
compute_variance(std::vector<double> const &);

inline double
compute_variance(std::vector<double> const &x)
{
  unsigned N = x.size();
  double mean = 0.0, variance = 0.0;

  for (unsigned k = 0; k < N; k++) { mean += x[k]; }
  mean /= N;
  for (unsigned k = 0; k < N; k++)
  {
    variance += (x[k] - mean)*(x[k] - mean);
  }
  return variance / (N-1);
}

#endif
