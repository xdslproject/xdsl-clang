#include <stdio.h>
#include <stdlib.h>

static void calc(int nx, int ny, int nz,
                 double *su, double *sv, double *sw,
                 double *u, double *v, double *w,
                 double *tzc1, double *tzc2, double *tzd1, double *tzd2) {
  printf("Initialise\n");

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        u[i * ny * nz + j * nz + k] = 10.0;
        v[i * ny * nz + j * nz + k] = 20.0;
        w[i * ny * nz + j * nz + k] = 30.0;
      }
    }
  }

  for (int k = 0; k < nz; k++) {
    tzc1[k] = 50.0;
    tzc2[k] = 15.0;
    tzd1[k] = 100.0;
    tzd2[k] = 5.0;
  }

  printf("Calculate\n");

  for (int i = 1; i < nx - 1; i++) {
    for (int j = 1; j < ny - 1; j++) {
      for (int k = 1; k < nz - 1; k++) {
        su[i * ny * nz + j * nz + k] =
            (2.0 * (u[(i - 1) * ny * nz + j * nz + k] *
                        (u[i * ny * nz + j * nz + k] +
                         u[(i - 1) * ny * nz + j * nz + k]) -
                    u[(i + 1) * ny * nz + j * nz + k] *
                        (u[i * ny * nz + j * nz + k] +
                         u[(i + 1) * ny * nz + j * nz + k]))) +
            (1.0 * (u[i * ny * nz + (j - 1) * nz + k] *
                        (v[i * ny * nz + (j - 1) * nz + k] +
                         v[(i + 1) * ny * nz + (j - 1) * nz + k]) -
                    u[i * ny * nz + (j + 1) * nz + k] *
                        (v[i * ny * nz + j * nz + k] +
                         v[(i + 1) * ny * nz + j * nz + k]))) +
            (tzc1[k] * u[i * ny * nz + j * nz + (k - 1)] *
                 (w[i * ny * nz + j * nz + (k - 1)] +
                  w[(i + 1) * ny * nz + j * nz + (k - 1)]) -
             tzc2[k] * u[i * ny * nz + j * nz + (k + 1)] *
                 (w[i * ny * nz + j * nz + k] +
                  w[(i + 1) * ny * nz + j * nz + k]));

        sv[i * ny * nz + j * nz + k] =
            (2.0 * (v[i * ny * nz + (j - 1) * nz + k] *
                        (v[i * ny * nz + j * nz + k] +
                         v[i * ny * nz + (j - 1) * nz + k]) -
                    v[i * ny * nz + (j + 1) * nz + k] *
                        (v[i * ny * nz + j * nz + k] +
                         v[i * ny * nz + (j + 1) * nz + k]))) +
            (2.0 * (v[(i - 1) * ny * nz + j * nz + k] *
                        (u[(i - 1) * ny * nz + j * nz + k] +
                         u[(i - 1) * ny * nz + (j + 1) * nz + k]) -
                    v[(i + 1) * ny * nz + j * nz + k] *
                        (u[i * ny * nz + j * nz + k] +
                         u[i * ny * nz + (j + 1) * nz + k]))) +
            (tzc1[k] * v[i * ny * nz + j * nz + (k - 1)] *
                 (w[i * ny * nz + j * nz + (k - 1)] +
                  w[i * ny * nz + (j + 1) * nz + (k - 1)]) -
             tzc2[k] * v[i * ny * nz + j * nz + (k + 1)] *
                 (w[i * ny * nz + j * nz + k] +
                  w[i * ny * nz + (j + 1) * nz + k]));

        sw[i * ny * nz + j * nz + k] =
            (tzd1[k] * w[i * ny * nz + j * nz + (k - 1)] *
                 (w[i * ny * nz + j * nz + k] +
                  w[i * ny * nz + j * nz + (k - 1)]) -
             tzd2[k] * w[i * ny * nz + j * nz + (k + 1)] *
                 (w[i * ny * nz + j * nz + k] +
                  w[i * ny * nz + j * nz + (k + 1)])) +
            (2.0 * (w[(i - 1) * ny * nz + j * nz + k] *
                        (u[(i - 1) * ny * nz + j * nz + k] +
                         u[(i - 1) * ny * nz + j * nz + (k + 1)]) -
                    w[(i + 1) * ny * nz + j * nz + k] *
                        (u[i * ny * nz + j * nz + k] +
                         u[i * ny * nz + j * nz + (k + 1)]))) +
            (2.0 * (w[i * ny * nz + (j - 1) * nz + k] *
                        (v[i * ny * nz + (j - 1) * nz + k] +
                         v[i * ny * nz + (j - 1) * nz + (k + 1)]) -
                    w[i * ny * nz + (j + 1) * nz + k] *
                        (v[i * ny * nz + j * nz + k] +
                         v[i * ny * nz + j * nz + (k + 1)])));
      }
    }
  }
}

static void wrapper(int nx, int ny, int nz) {
  double *su = malloc(nx * ny * nz * sizeof(double));
  double *sv = malloc(nx * ny * nz * sizeof(double));
  double *sw = malloc(nx * ny * nz * sizeof(double));
  double *u = malloc(nx * ny * nz * sizeof(double));
  double *v = malloc(nx * ny * nz * sizeof(double));
  double *w = malloc(nx * ny * nz * sizeof(double));
  double *tzc1 = malloc(nz * sizeof(double));
  double *tzc2 = malloc(nz * sizeof(double));
  double *tzd1 = malloc(nz * sizeof(double));
  double *tzd2 = malloc(nz * sizeof(double));

  calc(nx, ny, nz, su, sv, sw, u, v, w, tzc1, tzc2, tzd1, tzd2);

  free(su);
  free(sv);
  free(sw);
  free(u);
  free(v);
  free(w);
  free(tzc1);
  free(tzc2);
  free(tzd1);
  free(tzd2);
}

int main(void) {
  wrapper(512, 512, 512);
  return 0;
}
