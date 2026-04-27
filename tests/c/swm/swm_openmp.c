#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define L_OUT 1

static double cpu_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void run_model(int M, int N, int M_LEN, int N_LEN, int ITMAX) {
  float *u = malloc(N_LEN * M_LEN * sizeof(float));
  float *unew = malloc(N_LEN * M_LEN * sizeof(float));
  float *uold = malloc(N_LEN * M_LEN * sizeof(float));
  float *v = malloc(N_LEN * M_LEN * sizeof(float));
  float *vnew = malloc(N_LEN * M_LEN * sizeof(float));
  float *vold = malloc(N_LEN * M_LEN * sizeof(float));
  float *p = malloc(N_LEN * M_LEN * sizeof(float));
  float *pnew = malloc(N_LEN * M_LEN * sizeof(float));
  float *pold = malloc(N_LEN * M_LEN * sizeof(float));
  float *temp;

  float *cu = malloc(N_LEN * M_LEN * sizeof(float));
  float *cv = malloc(N_LEN * M_LEN * sizeof(float));
  float *z = malloc(N_LEN * M_LEN * sizeof(float));
  float *h = malloc(N_LEN * M_LEN * sizeof(float));
  float *psi = malloc(N_LEN * M_LEN * sizeof(float));

  float dt = 90.0f, tdt = dt;
  float dx = 1.0e5f, dy = 1.0e5f;
  float fsdx = 4.0f / dx, fsdy = 4.0f / dy;
  float a = 1.0e6f, alpha = 0.001f;
  float el = (float)N * dx;
  float pi = atan2f(0.0f, -1.0f);
  float tpi = pi + pi;
  float di = tpi / (float)M;
  float dj = tpi / (float)N;
  float pcf = pi * pi * a * a / (el * el);
  float tdts8, tdtsdx, tdtsdy;

  int mnmin = M < N ? M : N;
  float t100 = 0.0f, t200 = 0.0f, t300 = 0.0f;
  float time_v = 0.0f;
  double tstart, c1, c2;

  for (int j = 0; j < N_LEN; j++) {
    for (int i = 0; i < M_LEN; i++) {
      psi[j * M_LEN + i] = a * sinf((i + 0.5f) * di) * sinf((j + 0.5f) * dj);
      p[j * M_LEN + i] =
          pcf * (cosf(2.0f * i * di) + cosf(2.0f * j * dj)) + 50000.0f;
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      u[j * M_LEN + (i + 1)] =
          -(psi[(j + 1) * M_LEN + (i + 1)] - psi[j * M_LEN + (i + 1)]) / dy;
      v[(j + 1) * M_LEN + i] =
          (psi[(j + 1) * M_LEN + (i + 1)] - psi[(j + 1) * M_LEN + i]) / dx;
    }
  }

  for (int j = 0; j < N; j++) {
    u[j * M_LEN + 0] = u[j * M_LEN + (M_LEN - 1)];
    v[j * M_LEN + (M_LEN - 1)] = v[j * M_LEN + 0];
  }
  for (int i = 0; i < M; i++) {
    u[(N_LEN - 1) * M_LEN + i] = u[0 * M_LEN + i];
    v[0 * M_LEN + i] = v[(N_LEN - 1) * M_LEN + i];
  }
  u[(N_LEN - 1) * M_LEN + 0] = u[0 * M_LEN + (M_LEN - 1)];
  v[0 * M_LEN + (M_LEN - 1)] = v[(N_LEN - 1) * M_LEN + 0];

  for (int j = 0; j < N_LEN; j++) {
    for (int i = 0; i < M_LEN; i++) {
      uold[j * M_LEN + i] = u[j * M_LEN + i];
      vold[j * M_LEN + i] = v[j * M_LEN + i];
      pold[j * M_LEN + i] = p[j * M_LEN + i];
    }
  }

  if (L_OUT) {
    printf(" number of points in the x direction %d\n", N);
    printf(" number of points in the y direction %d\n", M);
    printf(" grid spacing in the x direction     %f\n", dx);
    printf(" grid spacing in the y direction     %f\n", dy);
    printf(" time step                           %f\n", dt);
    printf(" time filter parameter               %f\n", alpha);

    printf(" initial diagonal elements of p\n");
    for (int i = 0; i < mnmin; i++) printf("%f ", p[i * M_LEN + i]);
    printf("\n initial diagonal elements of u\n");
    for (int i = 0; i < mnmin; i++) printf("%f ", u[i * M_LEN + i]);
    printf("\n initial diagonal elements of v\n");
    for (int i = 0; i < mnmin; i++) printf("%f ", v[i * M_LEN + i]);
    printf("\n");
  }

  tstart = cpu_seconds();

  for (int ncycle = 1; ncycle <= ITMAX; ncycle++) {
    c1 = cpu_seconds();
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        cu[j * M_LEN + (i + 1)] =
            0.5f * (p[j * M_LEN + (i + 1)] + p[j * M_LEN + i]) *
            u[j * M_LEN + (i + 1)];
        cv[(j + 1) * M_LEN + i] =
            0.5f * (p[(j + 1) * M_LEN + i] + p[j * M_LEN + i]) *
            v[(j + 1) * M_LEN + i];
        z[(j + 1) * M_LEN + (i + 1)] =
            (fsdx * (v[(j + 1) * M_LEN + (i + 1)] - v[(j + 1) * M_LEN + i]) -
             fsdy * (u[(j + 1) * M_LEN + (i + 1)] - u[j * M_LEN + (i + 1)])) /
            (p[j * M_LEN + i] + p[j * M_LEN + (i + 1)] +
             p[(j + 1) * M_LEN + (i + 1)] + p[(j + 1) * M_LEN + i]);
        h[j * M_LEN + i] =
            p[j * M_LEN + i] +
            0.25f * (u[j * M_LEN + (i + 1)] * u[j * M_LEN + (i + 1)] +
                     u[j * M_LEN + i] * u[j * M_LEN + i] +
                     v[(j + 1) * M_LEN + i] * v[(j + 1) * M_LEN + i] +
                     v[j * M_LEN + i] * v[j * M_LEN + i]);
      }
    }
    c2 = cpu_seconds();
    t100 += (float)(c2 - c1);

    /* Periodic continuation */
    for (int j = 0; j < N; j++) {
      cu[j * M_LEN + 0] = cu[j * M_LEN + (M_LEN - 1)];
      cv[(j + 1) * M_LEN + (M_LEN - 1)] = cv[(j + 1) * M_LEN + 0];
      z[(j + 1) * M_LEN + 0] = z[(j + 1) * M_LEN + (M_LEN - 1)];
      h[j * M_LEN + (M_LEN - 1)] = h[j * M_LEN + 0];
    }
    for (int i = 0; i < M; i++) {
      cu[(N_LEN - 1) * M_LEN + (i + 1)] = cu[0 * M_LEN + (i + 1)];
      cv[0 * M_LEN + i] = cv[(N_LEN - 1) * M_LEN + i];
      z[0 * M_LEN + (i + 1)] = z[(N_LEN - 1) * M_LEN + (i + 1)];
      h[(N_LEN - 1) * M_LEN + i] = h[0 * M_LEN + i];
    }
    cu[(N_LEN - 1) * M_LEN + 0] = cu[0 * M_LEN + (M_LEN - 1)];
    cv[0 * M_LEN + (M_LEN - 1)] = cv[(N_LEN - 1) * M_LEN + 0];
    z[0 * M_LEN + 0] = z[(N_LEN - 1) * M_LEN + (M_LEN - 1)];
    h[(N_LEN - 1) * M_LEN + (M_LEN - 1)] = h[0 * M_LEN + 0];

    tdts8 = tdt / 8.0f;
    tdtsdx = tdt / dx;
    tdtsdy = tdt / dy;

    c1 = cpu_seconds();
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        unew[j * M_LEN + (i + 1)] =
            uold[j * M_LEN + (i + 1)] +
            tdts8 * (z[(j + 1) * M_LEN + (i + 1)] + z[j * M_LEN + (i + 1)]) *
                (cv[(j + 1) * M_LEN + (i + 1)] + cv[(j + 1) * M_LEN + i] +
                 cv[j * M_LEN + i] + cv[j * M_LEN + (i + 1)]) -
            tdtsdx * (h[j * M_LEN + (i + 1)] - h[j * M_LEN + i]);
        vnew[(j + 1) * M_LEN + i] =
            vold[(j + 1) * M_LEN + i] -
            tdts8 * (z[(j + 1) * M_LEN + (i + 1)] + z[(j + 1) * M_LEN + i]) *
                (cu[(j + 1) * M_LEN + (i + 1)] + cu[(j + 1) * M_LEN + i] +
                 cu[j * M_LEN + i] + cu[j * M_LEN + (i + 1)]) -
            tdtsdy * (h[(j + 1) * M_LEN + i] - h[j * M_LEN + i]);
        pnew[j * M_LEN + i] =
            pold[j * M_LEN + i] -
            tdtsdx * (cu[j * M_LEN + (i + 1)] - cu[j * M_LEN + i]) -
            tdtsdy * (cv[(j + 1) * M_LEN + i] - cv[j * M_LEN + i]);
      }
    }
    c2 = cpu_seconds();
    t200 += (float)(c2 - c1);

    for (int j = 0; j < N; j++) {
      unew[j * M_LEN + 0] = unew[j * M_LEN + (M_LEN - 1)];
      vnew[(j + 1) * M_LEN + (M_LEN - 1)] = vnew[(j + 1) * M_LEN + 0];
      pnew[j * M_LEN + (M_LEN - 1)] = pnew[j * M_LEN + 0];
    }
    for (int i = 0; i < M; i++) {
      unew[(N_LEN - 1) * M_LEN + (i + 1)] = unew[0 * M_LEN + (i + 1)];
      vnew[0 * M_LEN + i] = vnew[(N_LEN - 1) * M_LEN + i];
      pnew[(N_LEN - 1) * M_LEN + i] = pnew[0 * M_LEN + i];
    }
    unew[(N_LEN - 1) * M_LEN + 0] = unew[0 * M_LEN + (M_LEN - 1)];
    vnew[0 * M_LEN + (M_LEN - 1)] = vnew[(N_LEN - 1) * M_LEN + 0];
    pnew[(N_LEN - 1) * M_LEN + (M_LEN - 1)] = pnew[0 * M_LEN + 0];

    time_v += dt;
    if (ncycle > 1) {
      c1 = cpu_seconds();
      for (int j = 0; j < N_LEN; j++) {
        for (int i = 0; i < M_LEN; i++) {
          uold[j * M_LEN + i] = u[j * M_LEN + i] +
                                alpha * (unew[j * M_LEN + i] -
                                         2.0f * u[j * M_LEN + i] +
                                         uold[j * M_LEN + i]);
          vold[j * M_LEN + i] = v[j * M_LEN + i] +
                                alpha * (vnew[j * M_LEN + i] -
                                         2.0f * v[j * M_LEN + i] +
                                         vold[j * M_LEN + i]);
          pold[j * M_LEN + i] = p[j * M_LEN + i] +
                                alpha * (pnew[j * M_LEN + i] -
                                         2.0f * p[j * M_LEN + i] +
                                         pold[j * M_LEN + i]);
        }
      }

      temp = u; u = unew; unew = temp;
      temp = v; v = vnew; vnew = temp;
      temp = p; p = pnew; pnew = temp;
      c2 = cpu_seconds();
      t300 += (float)(c2 - c1);
    } else {
      tdt = tdt + tdt;
      for (int j = 0; j < N_LEN; j++) {
        for (int i = 0; i < N_LEN; i++) {
          uold[j * M_LEN + i] = u[j * M_LEN + i];
          vold[j * M_LEN + i] = v[j * M_LEN + i];
          pold[j * M_LEN + i] = p[j * M_LEN + i];
        }
      }
      temp = u; u = unew; unew = temp;
      temp = v; v = vnew; vnew = temp;
      temp = p; p = pnew; pnew = temp;
    }
  }

  temp = u; u = unew; unew = temp;
  temp = v; v = vnew; vnew = temp;
  temp = p; p = pnew; pnew = temp;

  if (L_OUT) {
    float ptime = time_v / 3600.0f;
    printf(" cycle number %d model time in hours %f\n", ITMAX, ptime);
    printf(" diagonal elements of p\n");
    for (int i = 0; i < mnmin; i++) printf("%f ", pnew[i * M_LEN + i]);
    printf("\n diagonal elements of u\n");
    for (int i = 0; i < mnmin; i++) printf("%f ", unew[i * M_LEN + i]);
    printf("\n diagonal elements of v\n");
    for (int i = 0; i < mnmin; i++) printf("%f ", vnew[i * M_LEN + i]);

    float mfs100 = 0, mfs200 = 0, mfs300 = 0;
    if (t100 > 0) mfs100 = (float)ITMAX * 24.0f * (float)(M * N) / t100 / 1.0e6f;
    if (t200 > 0) mfs200 = (float)ITMAX * 26.0f * (float)(M * N) / t200 / 1.0e6f;
    if (t300 > 0) mfs300 = (float)ITMAX * 15.0f * (float)(M * N) / t300 / 1.0e6f;

    double ctime = cpu_seconds() - tstart;
    double tcyc = ctime / (double)ITMAX;
    printf("\n cycle number %d total computer time %f time per cycle %f\n",
           ITMAX, ctime, tcyc);
    printf(" time and megaflops for loop 100 %f %f\n", t100, mfs100);
    printf(" time and megaflops for loop 200 %f %f\n", t200, mfs200);
    printf(" time and megaflops for loop 300 %f %f\n", t300, mfs300);
  }

  free(u); free(unew); free(uold);
  free(v); free(vnew); free(vold);
  free(p); free(pnew); free(pold);
  free(cu); free(cv); free(z); free(h); free(psi);
}

int main(void) {
  const int M = 512;
  const int N = 512;
  const int M_LEN = M + 1;
  const int N_LEN = N + 1;
  const int ITMAX = 4000;
  run_model(M, N, M_LEN, N_LEN, ITMAX);
  return 0;
}
