#include <stddef.h>
#include "../util/assertion.h"

static float global_array[100];

static void modify_array_one(int n, float *a, int idx, float value) {
  (void)n;
  a[idx - 1] = value;
}

static void modify_array_two(int n, float *a, int idx, float value) {
  (void)n;
  a[idx - 1] = value;
}

static void modify_3darray_one(int n1, int n2, int n3, int *arr,
                               int k, int j, int i, int value) {
  (void)n1;
  arr[(i - 1) * n2 * n3 + (j - 1) * n3 + (k - 1)] = value;
}

static void modify_3darray_two(int n1, int n2, int n3, int *arr,
                               int k, int j, int i, int value) {
  (void)n1;
  arr[(i - 1) * n2 * n3 + (j - 1) * n3 + (k - 1)] = value;
}

static void calc(void) {
  float a[100];
  float b[100];
  float z[10];
  float x[10];
  int c[15 * 5 * 10];

  int i, j, k;

  float z_init[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (i = 0; i < 10; i++) z[i] = z_init[i];
  for (i = 0; i < 10; i++) x[i] = (float)(i + 11);

  for (i = 0; i < 10; i++) {
    ASSERT(z[i] == (float)(i + 1));
    ASSERT(x[i] == (float)(i + 11));
  }

  ASSERT(sizeof(a) / sizeof(a[0]) == 100);
  ASSERT(sizeof(b) / sizeof(b[0]) == 100);
  ASSERT(sizeof(global_array) / sizeof(global_array[0]) == 100);
  ASSERT(sizeof(c) / sizeof(int) == 750);

  for (i = 0; i < 100; i++) {
    a[i] = (float)(i + 1);
    b[i] = (float)(99 - i);
    global_array[i] = (float)((i + 1) * 10);
  }

  for (i = 0; i < 100; i++) {
    ASSERT(a[i] == (float)(i + 1));
    ASSERT(b[i] == (float)(99 - i));
    ASSERT(global_array[i] == (float)((i + 1) * 10));
  }

  a[19] = 34.5f;
  b[49] = 165.2f;
  global_array[69] = 23.1f;
  ASSERT(a[19] == 34.5f);
  ASSERT(b[49] == 165.2f);
  ASSERT(global_array[69] == 23.1f);

  modify_array_one(100, a, 20, 20.0f);
  modify_array_two(100, b, 50, 50.0f);
  modify_array_two(100, global_array, 70, 700.0f);
  for (i = 0; i < 100; i++) {
    ASSERT(a[i] == (float)(i + 1));
    ASSERT(b[i] == (float)(99 - i));
    ASSERT(global_array[i] == (float)((i + 1) * 10));
  }

  modify_array_one(100, global_array, 60, 123.4f);
  ASSERT(global_array[59] == 123.4f);
  global_array[59] = 600.0f;

  for (i = 0; i < 100; i++) a[i] = b[i];
  for (i = 0; i < 100; i++) b[i] = global_array[i];
  for (i = 0; i < 100; i++) {
    ASSERT(a[i] == (float)(99 - i));
    ASSERT(b[i] == (float)((i + 1) * 10));
  }

  for (i = 0; i < 15; i++)
    for (j = 0; j < 5; j++)
      for (k = 0; k < 10; k++)
        c[i * 5 * 10 + j * 10 + k] = (k + 1) + ((j + 1) * 10) + ((i + 1) * 100);

  ASSERT(c[(5 - 1) * 5 * 10 + (4 - 1) * 10 + (3 - 1)] == 543);
  ASSERT(c[(15 - 1) * 5 * 10 + (5 - 1) * 10 + (8 - 1)] == 1558);

  modify_3darray_one(15, 5, 10, c, 2, 3, 4, 100);
  ASSERT(c[(4 - 1) * 5 * 10 + (3 - 1) * 10 + (2 - 1)] == 100);
  modify_3darray_two(15, 5, 10, c, 6, 2, 12, 200);
  ASSERT(c[(12 - 1) * 5 * 10 + (2 - 1) * 10 + (6 - 1)] == 200);
}

int main(void) {
  assert_init(0);
  calc();
  assert_finalize(__FILE__);
  return 0;
}
