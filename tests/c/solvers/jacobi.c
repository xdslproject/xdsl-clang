#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define LEFT_VALUE 1.0
#define RIGHT_VALUE 10.0
#define MAX_ITERATIONS 100000
#define REPORT_NORM_PERIOD 100

static void initialise_values(int nx, int ny, double *u_k, double *u_kp1) {
  int row = ny + 2;
  for (int i = 0; i < nx + 2; i++) {
    for (int j = 1; j <= ny; j++) u_k[i * row + j] = 0.0;
  }
  for (int i = 0; i < nx + 2; i++) {
    u_k[i * row + 0] = LEFT_VALUE;
    u_k[i * row + (ny + 1)] = RIGHT_VALUE;
  }
  for (int i = 0; i < nx + 2; i++) {
    for (int j = 0; j < ny + 2; j++) u_kp1[i * row + j] = u_k[i * row + j];
  }
}

static void run_solver(int nx, int ny, double convergence_accuracy) {
  int row = ny + 2;
  double *u_k = malloc((nx + 2) * row * sizeof(double));
  double *u_kp1 = malloc((nx + 2) * row * sizeof(double));
  double *temp;
  double bnorm = 0.0, rnorm = 0.0, norm = 0.0;
  int k = 0;

  printf("Global size in X= %d Global size in Y= %d\n", nx, ny);

  initialise_values(nx, ny, u_k, u_kp1);

  for (int i = 1; i <= nx; i++) {
    for (int j = 1; j <= ny; j++) {
      double r = u_k[i * row + j] * 4 - u_k[i * row + (j - 1)] -
                 u_k[i * row + (j + 1)] - u_k[(i - 1) * row + j] -
                 u_k[(i + 1) * row + j];
      bnorm += r * r;
    }
  }
  bnorm = sqrt(bnorm);

  for (k = 0; k <= MAX_ITERATIONS; k++) {
    rnorm = 0.0;
    for (int i = 1; i <= nx; i++) {
      for (int j = 1; j <= ny; j++) {
        double r = u_k[i * row + j] * 4 - u_k[i * row + (j - 1)] -
                   u_k[i * row + (j + 1)] - u_k[(i - 1) * row + j] -
                   u_k[(i + 1) * row + j];
        rnorm += r * r;
      }
    }
    norm = sqrt(rnorm) / bnorm;
    if (norm < convergence_accuracy) break;

    for (int i = 1; i <= nx; i++) {
      for (int j = 1; j <= ny; j++) {
        u_kp1[i * row + j] = 0.25 * (u_k[i * row + (j - 1)] +
                                      u_k[i * row + (j + 1)] +
                                      u_k[(i - 1) * row + j] +
                                      u_k[(i + 1) * row + j]);
      }
    }

    temp = u_kp1;
    u_kp1 = u_k;
    u_k = temp;

    if (k % REPORT_NORM_PERIOD == 0)
      printf("Iteration= %d Relative Norm= %e\n", k, norm);
  }
  printf("Terminated on %d iterations, Relative Norm= %e\n", k, norm);

  free(u_k);
  free(u_kp1);
}

int main(void) {
  run_solver(512, 512, 1e-4);
  return 0;
}
