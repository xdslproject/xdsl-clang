#include <stdlib.h>
#include "../util/assertion.h"

static float global_array[6 * 10];

static void calc(int j) {
  float a[100];
  float b[100];
  float cc[100];
  float *n = malloc(6 * 10 * sizeof(float));
  float *m = malloc(6 * 10 * sizeof(float));
  float p[6 * 10];
  float q[6 * 10];

  int i, k;

  for (i = 0; i < 100; i++) a[i] = (float)((i + 1) + j);
  for (i = 0; i < 100; i++) b[i] = 20.0f;
  for (i = 0; i < 100; i++) cc[i] = a[i] + b[i];

  for (i = 0; i < 100; i++) {
    ASSERT(a[i] == (float)((i + 1) + j));
    ASSERT(b[i] == 20.0f);
    ASSERT(cc[i] == a[i] + b[i]);
  }

  for (i = 0; i < 100; i++) cc[i] = a[i] + b[i] + 100;
  for (i = 0; i < 100; i++) ASSERT(cc[i] == a[i] + b[i] + 100);

  for (k = 0; k < 6; k++)
    for (i = 0; i < 10; i++) {
      n[k * 10 + i] = 100.0f;
      p[k * 10 + i] = 100.0f;
    }

  for (k = 0; k < 6; k++)
    for (i = 0; i < 10; i++)
      m[k * 10 + i] = n[k * 10 + i] - p[k * 10 + i];

  for (i = 0; i < 10; i++)
    for (k = 0; k < 6; k++)
      ASSERT(m[k * 10 + i] == 0.0f);

  for (i = 0; i < 10; i++)
    for (k = 0; k < 6; k++) {
      m[k * 10 + i] = (float)(i + 1);
      p[k * 10 + i] = (float)(k + 1);
    }

  for (k = 0; k < 6; k++)
    for (i = 0; i < 10; i++) {
      q[k * 10 + i] = m[k * 10 + i] * p[k * 10 + i];
      n[k * 10 + i] = m[k * 10 + i] - p[k * 10 + i];
    }

  for (i = 0; i < 10; i++)
    for (k = 0; k < 6; k++) {
      ASSERT(q[k * 10 + i] == (float)((i + 1) * (k + 1)));
      ASSERT(n[k * 10 + i] == (float)(i - k));
    }

  for (k = 0; k < 6; k++)
    for (i = 0; i < 10; i++)
      global_array[k * 10 + i] = m[k * 10 + i] * p[k * 10 + i] * 10 + n[k * 10 + i];

  for (i = 0; i < 10; i++)
    for (k = 0; k < 6; k++)
      ASSERT(global_array[k * 10 + i] == (float)((i + 1) * (k + 1) * 10 + (i - k)));

  free(n);
  free(m);
}

int main(void) {
  assert_init(0);
  calc(10);
  assert_finalize(__FILE__);
  return 0;
}
