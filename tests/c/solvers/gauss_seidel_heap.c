#include <stdlib.h>

static void run(void) {
  /* data dims (M, J, I) = (1024, 512, 512), C row-major: data[m*512*512 + j*512 + i] */
  const int dim_m = 1024;
  const int dim_j = 512;
  const int dim_i = 512;
  float *data = malloc(dim_m * dim_j * dim_i * sizeof(float));

  for (int kk = 1; kk <= 1; kk++) {
    for (int m = 1; m < dim_m - 1; m++) {
      for (int j = 1; j < dim_j - 1; j++) {
        for (int i = 1; i < dim_i - 1; i++) {
          data[m * dim_j * dim_i + j * dim_i + i] =
              (data[m * dim_j * dim_i + (j - 1) * dim_i + i] +
               data[m * dim_j * dim_i + (j + 1) * dim_i + i] +
               data[m * dim_j * dim_i + j * dim_i + (i - 1)] +
               data[m * dim_j * dim_i + j * dim_i + (i + 1)] +
               data[(m - 1) * dim_j * dim_i + j * dim_i + i] +
               data[(m + 1) * dim_j * dim_i + j * dim_i + i]) *
              0.16666f;
        }
      }
    }
  }

  free(data);
}

int main(void) {
  run();
  return 0;
}
