/*
 * Smaller-fixture variant of tra_adv.c — same NEMO advection kernel logic
 * with reduced grid dimensions and itn_count so it fits inside the default
 * per-test runtime budget. The full-size tra_adv.c stays around for
 * reference / nightly runs but is xfailed under the harness's 60s budget
 * (Task F4).
 *
 * Original kernel: traadv from the NEMO software (http://www.nemo-ocean.eu),
 * governed by the CeCILL licence (http://www.cecill.info), IS-ENES2 - CMCC/STFC.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double sign(double a, double b) {
  double mag = fabs(a);
  return (b >= 0.0) ? mag : -mag;
}

static void tra_adv(void) {
  const int jpi = 16;
  const int jpj = 8;
  const int jpk = 8;
  const long itn_count = 2;

  /* 3D arrays laid out flat row-major as [jpk][jpj][jpi]. */
  double *tsn = malloc(jpk * jpj * jpi * sizeof(double));
  double *pun = malloc(jpk * jpj * jpi * sizeof(double));
  double *pvn = malloc(jpk * jpj * jpi * sizeof(double));
  double *pwn = malloc(jpk * jpj * jpi * sizeof(double));
  double *mydomain = malloc(jpk * jpj * jpi * sizeof(double));
  double *zslpx = malloc(jpk * jpj * jpi * sizeof(double));
  double *zslpy = malloc(jpk * jpj * jpi * sizeof(double));
  double *zwx = malloc(jpk * jpj * jpi * sizeof(double));
  double *zwy = malloc(jpk * jpj * jpi * sizeof(double));
  double *umask = malloc(jpk * jpj * jpi * sizeof(double));
  double *vmask = malloc(jpk * jpj * jpi * sizeof(double));
  double *tmask = malloc(jpk * jpj * jpi * sizeof(double));
  double *zind = malloc(jpk * jpj * jpi * sizeof(double));

  /* 2D arrays laid out flat row-major as [jpj][jpi]. */
  double *ztfreez = malloc(jpj * jpi * sizeof(double));
  double *rnfmsk = malloc(jpj * jpi * sizeof(double));
  double *upsmsk = malloc(jpj * jpi * sizeof(double));
  double *rnfmsk_z = malloc(jpk * sizeof(double));

  double r = (double)jpi * jpj * jpk;

  printf("Initialising\n");

  for (int jk = 0; jk < jpk; jk++) {
    for (int jj = 0; jj < jpj; jj++) {
      for (int ji = 0; ji < jpi; ji++) {
        double v = (double)((ji + 1) * (jj + 1) * (jk + 1)) / r;
        umask[jk * jpj * jpi + jj * jpi + ji] = v;
        mydomain[jk * jpj * jpi + jj * jpi + ji] = v;
        pun[jk * jpj * jpi + jj * jpi + ji] = v;
        pvn[jk * jpj * jpi + jj * jpi + ji] = v;
        pwn[jk * jpj * jpi + jj * jpi + ji] = v;
        vmask[jk * jpj * jpi + jj * jpi + ji] = v;
        tsn[jk * jpj * jpi + jj * jpi + ji] = v;
        tmask[jk * jpj * jpi + jj * jpi + ji] = v;
      }
    }
  }

  r = (double)jpi * jpj;
  for (int jj = 0; jj < jpj; jj++) {
    for (int ji = 0; ji < jpi; ji++) {
      double v = (double)((ji + 1) * (jj + 1)) / r;
      ztfreez[jj * jpi + ji] = v;
      upsmsk[jj * jpi + ji] = v;
      rnfmsk[jj * jpi + ji] = v;
    }
  }

  for (int jk = 0; jk < jpk; jk++) rnfmsk_z[jk] = (double)(jk + 1) / jpk;

  for (long jt = 1; jt <= itn_count; jt++) {
    printf("Starting iteration %ld\n", jt);

    for (int jk = 0; jk < jpk; jk++) {
      for (int jj = 0; jj < jpj; jj++) {
        for (int ji = 0; ji < jpi; ji++) {
          double a = rnfmsk[jj * jpi + ji] * rnfmsk_z[jk];
          double b = upsmsk[jj * jpi + ji];
          double m = (a > b) ? a : b;
          zind[jk * jpj * jpi + jj * jpi + ji] =
              1.0 - m * tmask[jk * jpj * jpi + jj * jpi + ji];
        }
      }
    }

    for (int jj = 0; jj < jpj; jj++) {
      for (int ji = 0; ji < jpi; ji++) {
        zwx[(jpk - 1) * jpj * jpi + jj * jpi + ji] = 0.0;
        zwy[(jpk - 1) * jpj * jpi + jj * jpi + ji] = 0.0;
      }
    }

    for (int jk = 0; jk < jpk - 1; jk++) {
      for (int jj = 0; jj < jpj - 1; jj++) {
        for (int ji = 0; ji < jpi - 1; ji++) {
          zwx[jk * jpj * jpi + jj * jpi + ji] =
              umask[jk * jpj * jpi + jj * jpi + ji] *
              (mydomain[jk * jpj * jpi + jj * jpi + (ji + 1)] -
               mydomain[jk * jpj * jpi + jj * jpi + ji]);
          zwy[jk * jpj * jpi + jj * jpi + ji] =
              vmask[jk * jpj * jpi + jj * jpi + ji] *
              (mydomain[jk * jpj * jpi + (jj + 1) * jpi + ji] -
               mydomain[jk * jpj * jpi + jj * jpi + ji]);
        }
      }
    }

    for (int jj = 0; jj < jpj; jj++) {
      for (int ji = 0; ji < jpi; ji++) {
        zslpx[(jpk - 1) * jpj * jpi + jj * jpi + ji] = 0.0;
        zslpy[(jpk - 1) * jpj * jpi + jj * jpi + ji] = 0.0;
      }
    }

    for (int jk = 0; jk < jpk - 1; jk++) {
      for (int jj = 1; jj < jpj; jj++) {
        for (int ji = 1; ji < jpi; ji++) {
          zslpx[jk * jpj * jpi + jj * jpi + ji] =
              (zwx[jk * jpj * jpi + jj * jpi + ji] +
               zwx[jk * jpj * jpi + jj * jpi + (ji - 1)]) *
              (0.25 + sign(0.25, zwx[jk * jpj * jpi + jj * jpi + ji] *
                                     zwx[jk * jpj * jpi + jj * jpi + (ji - 1)]));
          zslpy[jk * jpj * jpi + jj * jpi + ji] =
              (zwy[jk * jpj * jpi + jj * jpi + ji] +
               zwy[jk * jpj * jpi + (jj - 1) * jpi + ji]) *
              (0.25 + sign(0.25, zwy[jk * jpj * jpi + jj * jpi + ji] *
                                     zwy[jk * jpj * jpi + (jj - 1) * jpi + ji]));
        }
      }
    }

    for (int jk = 0; jk < jpk - 1; jk++) {
      for (int jj = 1; jj < jpj; jj++) {
        for (int ji = 1; ji < jpi; ji++) {
          double ax = fabs(zslpx[jk * jpj * jpi + jj * jpi + ji]);
          double bx = 2.0 * fabs(zwx[jk * jpj * jpi + jj * jpi + (ji - 1)]);
          double cx = 2.0 * fabs(zwx[jk * jpj * jpi + jj * jpi + ji]);
          double mx = ax < bx ? ax : bx;
          if (cx < mx) mx = cx;
          zslpx[jk * jpj * jpi + jj * jpi + ji] =
              sign(1.0, zslpx[jk * jpj * jpi + jj * jpi + ji]) * mx;

          double ay = fabs(zslpy[jk * jpj * jpi + jj * jpi + ji]);
          double by = 2.0 * fabs(zwy[jk * jpj * jpi + (jj - 1) * jpi + ji]);
          double cy = 2.0 * fabs(zwy[jk * jpj * jpi + jj * jpi + ji]);
          double my = ay < by ? ay : by;
          if (cy < my) my = cy;
          zslpy[jk * jpj * jpi + jj * jpi + ji] =
              sign(1.0, zslpy[jk * jpj * jpi + jj * jpi + ji]) * my;
        }
      }
    }

    for (int jk = 0; jk < jpk - 1; jk++) {
      for (int jj = 1; jj < jpj - 1; jj++) {
        for (int ji = 1; ji < jpi - 1; ji++) {
          double pu = pun[jk * jpj * jpi + jj * jpi + ji];
          double pv = pvn[jk * jpj * jpi + jj * jpi + ji];
          double zin = zind[jk * jpj * jpi + jj * jpi + ji];

          zwx[jk * jpj * jpi + jj * jpi + ji] =
              pu * ((0.5 - sign(0.5, pu)) *
                        mydomain[jk * jpj * jpi + jj * jpi + (ji + 1)] +
                    zin * ((sign(0.5, pu) - 0.5 * pu) *
                           zslpx[jk * jpj * jpi + jj * jpi + (ji + 1)]) +
                    (1.0 - (0.5 - sign(0.5, pu))) *
                        mydomain[jk * jpj * jpi + jj * jpi + ji] +
                    zin * ((sign(0.5, pu) - 0.5 * pu) *
                           zslpx[jk * jpj * jpi + jj * jpi + ji]));

          zwy[jk * jpj * jpi + jj * jpi + ji] =
              pv * ((0.5 - sign(0.5, pv)) *
                        mydomain[jk * jpj * jpi + (jj + 1) * jpi + ji] +
                    zin * ((sign(0.5, pv) - 0.5 * pv) *
                           zslpy[jk * jpj * jpi + (jj + 1) * jpi + ji]) +
                    (1.0 - (0.5 - sign(0.5, pv))) *
                        mydomain[jk * jpj * jpi + jj * jpi + ji] +
                    zin * ((sign(0.5, pv) - 0.5 * pv) *
                           zslpy[jk * jpj * jpi + jj * jpi + ji]));
        }
      }
    }

    for (int jk = 0; jk < jpk - 1; jk++) {
      for (int jj = 1; jj < jpj - 1; jj++) {
        for (int ji = 1; ji < jpi - 1; ji++) {
          mydomain[jk * jpj * jpi + jj * jpi + ji] +=
              -1.0 * (zwx[jk * jpj * jpi + jj * jpi + ji] -
                      zwx[jk * jpj * jpi + jj * jpi + (ji - 1)] +
                      zwy[jk * jpj * jpi + jj * jpi + ji] -
                      zwy[jk * jpj * jpi + (jj - 1) * jpi + ji]);
        }
      }
    }

    for (int jj = 0; jj < jpj; jj++) {
      for (int ji = 0; ji < jpi; ji++) {
        zwx[0 * jpj * jpi + jj * jpi + ji] = 0.0;
        zwx[(jpk - 1) * jpj * jpi + jj * jpi + ji] = 0.0;
      }
    }

    for (int jk = 1; jk < jpk - 1; jk++) {
      for (int jj = 0; jj < jpj; jj++) {
        for (int ji = 0; ji < jpi; ji++) {
          zwx[jk * jpj * jpi + jj * jpi + ji] =
              tmask[jk * jpj * jpi + jj * jpi + ji] *
              (mydomain[(jk - 1) * jpj * jpi + jj * jpi + ji] -
               mydomain[jk * jpj * jpi + jj * jpi + ji]);
        }
      }
    }

    for (int jj = 0; jj < jpj; jj++) {
      for (int ji = 0; ji < jpi; ji++)
        zslpx[0 * jpj * jpi + jj * jpi + ji] = 0.0;
    }

    for (int jk = 1; jk < jpk - 1; jk++) {
      for (int jj = 0; jj < jpj; jj++) {
        for (int ji = 0; ji < jpi; ji++) {
          zslpx[jk * jpj * jpi + jj * jpi + ji] =
              (zwx[jk * jpj * jpi + jj * jpi + ji] +
               zwx[(jk + 1) * jpj * jpi + jj * jpi + ji]) *
              (0.25 + sign(0.25, zwx[jk * jpj * jpi + jj * jpi + ji] *
                                     zwx[(jk + 1) * jpj * jpi + jj * jpi + ji]));
        }
      }
    }

    for (int jk = 1; jk < jpk - 1; jk++) {
      for (int jj = 0; jj < jpj; jj++) {
        for (int ji = 0; ji < jpi; ji++) {
          double a = fabs(zslpx[jk * jpj * jpi + jj * jpi + ji]);
          double b = 2.0 * fabs(zwx[(jk + 1) * jpj * jpi + jj * jpi + ji]);
          double c = 2.0 * fabs(zwx[jk * jpj * jpi + jj * jpi + ji]);
          double m = a < b ? a : b;
          if (c < m) m = c;
          zslpx[jk * jpj * jpi + jj * jpi + ji] =
              sign(1.0, zslpx[jk * jpj * jpi + jj * jpi + ji]) * m;
        }
      }
    }

    for (int jj = 0; jj < jpj; jj++) {
      for (int ji = 0; ji < jpi; ji++) {
        zwx[0 * jpj * jpi + jj * jpi + ji] =
            pwn[0 * jpj * jpi + jj * jpi + ji] *
            mydomain[0 * jpj * jpi + jj * jpi + ji];
      }
    }

    for (int jk = 0; jk < jpk - 1; jk++) {
      for (int jj = 1; jj < jpj - 1; jj++) {
        for (int ji = 1; ji < jpi - 1; ji++) {
          double pw = pwn[(jk + 1) * jpj * jpi + jj * jpi + ji];
          double zin = zind[jk * jpj * jpi + jj * jpi + ji];
          zwx[(jk + 1) * jpj * jpi + jj * jpi + ji] =
              pw * (0.5 + sign(0.5, pw) *
                              (mydomain[(jk + 1) * jpj * jpi + jj * jpi + ji] +
                               zin * (sign(0.5, pw) -
                                      0.5 * pw *
                                          zslpx[(jk + 1) * jpj * jpi + jj * jpi + ji])) +
                    (1.0 - 0.5 + sign(0.5, pw)) *
                        (mydomain[jk * jpj * jpi + jj * jpi + ji] +
                         zin * (sign(0.5, pw) -
                                0.5 * pw *
                                    zslpx[jk * jpj * jpi + jj * jpi + ji])));
        }
      }
    }

    for (int jk = 0; jk < jpk - 1; jk++) {
      for (int jj = 1; jj < jpj - 1; jj++) {
        for (int ji = 1; ji < jpi - 1; ji++) {
          mydomain[jk * jpj * jpi + jj * jpi + ji] =
              -1.0 * (zwx[jk * jpj * jpi + jj * jpi + ji] -
                      zwx[(jk + 1) * jpj * jpi + jj * jpi + ji]);
        }
      }
    }
  }

  free(tsn);
  free(pun);
  free(pvn);
  free(pwn);
  free(mydomain);
  free(zslpx);
  free(zslpy);
  free(zwx);
  free(zwy);
  free(umask);
  free(vmask);
  free(tmask);
  free(zind);
  free(ztfreez);
  free(rnfmsk);
  free(upsmsk);
  free(rnfmsk_z);
}

int main(void) {
  tra_adv();
  return 0;
}
