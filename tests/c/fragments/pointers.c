#include <stdlib.h>
#include "../util/assertion.h"

static float *ptr3 = NULL;

static void swap(float **swp1, float **swp2) {
  float *t = *swp1;
  *swp1 = *swp2;
  *swp2 = t;
}

static void modify_array_one(float *a, int idx, float value) {
  a[idx] = value;
}

static void modify_3darray_one(int n2, int n3, int *arr,
                               int k, int j, int i, int value) {
  arr[i * n2 * n3 + j * n3 + k] = value;
}

static void calc(void) {
  float *a = malloc(100 * sizeof(float));
  float *b = malloc(100 * sizeof(float));
  float *z = malloc(10 * sizeof(float));
  float *x = malloc(10 * sizeof(float));
  int *c = malloc(15 * 5 * 10 * sizeof(int));
  float *ptr1 = NULL, *ptr2 = NULL;
  int *ptr_md = NULL;
  float t;

  ptr1 = z;
  ptr2 = x;

  float z_init[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (int i = 0; i < 10; i++) ptr1[i] = z_init[i];
  for (int i = 0; i < 10; i++) ptr2[i] = (float)(i + 11);

  for (int i = 0; i < 10; i++) {
    ASSERT(ptr1[i] == (float)(i + 1));
    ASSERT(z[i] == (float)(i + 1));
    ASSERT(ptr2[i] == (float)(i + 11));
    ASSERT(x[i] == (float)(i + 11));
  }

  /* ptr2 = ptr1 — whole-array assignment, copies values */
  for (int i = 0; i < 10; i++) ptr2[i] = ptr1[i];
  for (int i = 0; i < 10; i++) {
    ASSERT(ptr2[i] == (float)(i + 1));
    ASSERT(x[i] == (float)(i + 1));
  }

  for (int i = 0; i < 100; i++) {
    a[i] = (float)(i + 1);
    b[i] = (float)(99 - i);
  }

  ptr1 = a;
  for (int i = 0; i < 100; i++) ASSERT(ptr1[i] == (float)(i + 1));

  ptr2 = a;
  for (int i = 0; i < 100; i++) ASSERT(ptr2[i] == (float)(i + 1));

  ptr3 = ptr1;
  for (int i = 0; i < 100; i++) ASSERT(ptr3[i] == (float)(i + 1));

  ptr1[19] = 34.0f;
  ASSERT(a[19] == 34.0f);
  ASSERT(ptr1[19] == 34.0f);
  ASSERT(ptr2[19] == 34.0f);
  ASSERT(ptr3[19] == 34.0f);

  ptr2 = b;
  for (int i = 0; i < 100; i++) {
    if (i == 19) ASSERT(ptr1[i] == 34.0f);
    else ASSERT(ptr1[i] == (float)(i + 1));
    ASSERT(ptr2[i] == (float)(99 - i));
  }

  swap(&ptr1, &ptr2);
  for (int i = 0; i < 100; i++) {
    if (i == 19) ASSERT(ptr2[i] == 34.0f);
    else ASSERT(ptr2[i] == (float)(i + 1));
    ASSERT(ptr1[i] == (float)(99 - i));
  }

  t = ptr1[2];
  ASSERT(t == (float)(100 - 3));

  modify_array_one(ptr1, 19, 3.1f);
  ASSERT(ptr1[19] == 3.1f);

  modify_array_one(ptr1, 18, 87.64f);
  ASSERT(ptr1[18] == 87.64f);

  modify_array_one(ptr1, 75, 992.32f);
  ASSERT(ptr1[75] == 992.32f);

  for (int i = 0; i < 15; i++)
    for (int j = 0; j < 5; j++)
      for (int k = 0; k < 10; k++)
        c[i * 5 * 10 + j * 10 + k] = (k + 1) + ((j + 1) * 10) + ((i + 1) * 100);

  ptr_md = c;
  ASSERT(ptr_md[(5 - 1) * 5 * 10 + (4 - 1) * 10 + (3 - 1)] == 543);
  ASSERT(ptr_md[(15 - 1) * 5 * 10 + (5 - 1) * 10 + (8 - 1)] == 1558);

  modify_3darray_one(5, 10, ptr_md, 1, 2, 3, 100);
  ASSERT(ptr_md[3 * 5 * 10 + 2 * 10 + 1] == 100);

  modify_3darray_one(5, 10, ptr_md, 3, 3, 6, 87);
  ASSERT(ptr_md[6 * 5 * 10 + 3 * 10 + 3] == 87);

  modify_3darray_one(5, 10, ptr_md, 6, 0, 2, 13);
  ASSERT(ptr_md[2 * 5 * 10 + 0 * 10 + 6] == 13);

  free(a);
  free(b);
  free(c);
  free(z);
  free(x);
}

int main(void) {
  assert_init(0);
  calc();
  assert_finalize(__FILE__);
  return 0;
}
