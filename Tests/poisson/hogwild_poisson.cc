/* 
 * benchmark_hogwild_regression.cpp
 * author: Abhijit Chowdhary (achowdh2@ncsu.edu)
 *
 * Benchmark HOGWILD! as applied to solving the 1d Poisson equation:
 *  u''(x) = -sin(x), u(0) = 0, u(2*pi) = 0.
 * using the standard regression loss function:
 *  minimize (1/2)||Ax-b||_2^2
 *
 * Outputs to results.txt the vector approximating the solution to the Poisson
 * equation and to stdout the time taken.
 */
#include <omp.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <random>

#define ETA 0.01
#define NUM_EPOCHS 20
#define TRIALS_PER_CORE 10

int 
main(int argc, char **argv)
{
  omp_set_dynamic(0);
  auto rng = std::default_random_engine {}; rng.seed(0);
  double time_taken = 0.0;

  unsigned P = omp_get_max_threads();

  // Read MSD dataset into memory and format matrices.
  const unsigned N = 1 << 7;
  double dx = 2*M_PI / (N-1); double dx2 = dx*dx;

  double f[N]; std::atomic<double> x[N];
  for (int k = 0; k < N; k++) 
  { 
    f[k] = -sin(k*dx); 
  }
  
  std::array<unsigned, N> ordering;
  std::iota(ordering.begin(), ordering.end(), 0);

  for (unsigned trial = 0; trial < TRIALS_PER_CORE; ++trial)
  {
    for (unsigned k = 0; k < N; k++) { x[k] = 0.0; }

    double t_start, t_end;
    t_start = omp_get_wtime();
    for (unsigned epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
      std::shuffle(ordering.begin(), ordering.end(), rng);
      #pragma omp parallel for
      for (unsigned k = 0; k < N; k++)
      {
        unsigned id = ordering[k];

        double dg = -2*x[id].load()/dx2 - f[id];
        if (id != 0) { dg += x[id-1].load()/dx2; }
        if (id != N-1) { dg += x[id+1].load()/dx2; }

        x[id].exchange(   x[id].load() - ETA*(-2*dg/dx2) );
        if (id != 0) { x[id-1].exchange( x[id-1].load() - ETA*dg/dx2 ); }
        if (id != N-1) { x[id+1].exchange( x[id+1].load() - ETA*dg/dx2 ); }

      }
      printf("%f\n", x[N/2].load());
    }
    t_end = omp_get_wtime();
    time_taken += t_end-t_start;
  }
  time_taken /= TRIALS_PER_CORE;

  printf("Time taken: %.5f\n", time_taken);
  FILE *file = fopen("results.txt", "wb");
  for (unsigned k = 0; k < N; k++) { fprintf(file, "%f\n", x[k].load()); }
  fclose(file);

  return 0;
}
