#include <stdlib.h>
#include "../util/assertion.h"

static void test_transpose(void) {
  int a[10 * 10], b[10 * 10];
  int *c = malloc(10 * 10 * sizeof(int));
  int *d = malloc(10 * 10 * sizeof(int));

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      a[i * 10 + j] = j + 1;
      c[i * 10 + j] = j + 1;
    }
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      b[i * 10 + j] = a[j * 10 + i];
      d[i * 10 + j] = c[j * 10 + i];
    }
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ASSERT(b[i * 10 + j] == i + 1);
      ASSERT(d[i * 10 + j] == i + 1);
    }
  }

  free(c);
  free(d);
}

static void compute_matmul(int rows, int inner, int cols,
                           int n1, int n2, int n3,
                           float *a, float *b, float *c) {
  (void)n1; (void)n2; (void)n3;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float s = 0.0f;
      for (int k = 0; k < inner; k++) s += a[i * inner + k] * b[k * cols + j];
      c[i * cols + j] = s;
    }
  }
}

static void compare_matmul(int rows, int inner, int cols,
                           float *a, float *b, float *c) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float comp = 0.0f;
      for (int k = 0; k < inner; k++) comp += a[i * inner + k] * b[k * cols + j];
      ASSERT(c[i * cols + j] == comp);
    }
  }
}

static void test_matmul(void) {
  float a[5 * 5], b[5 * 5], c[5 * 5];

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      a[i * 5 + j] = (float)(j + 1);
      b[i * 5 + j] = (float)(i + 1);
    }
  }
  compute_matmul(5, 5, 5, 0, 0, 0, a, b, c);
  compare_matmul(5, 5, 5, a, b, c);

  float *d = malloc(10 * 10 * sizeof(float));
  float *e = malloc(10 * 10 * sizeof(float));
  float *f = malloc(10 * 10 * sizeof(float));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      d[i * 10 + j] = (float)(j + 1);
      e[i * 10 + j] = (float)(i + 1);
    }
  }
  compute_matmul(10, 10, 10, 0, 0, 0, d, e, f);
  compare_matmul(10, 10, 10, d, e, f);
  free(d);
  free(e);
  free(f);
}

static void test_sum(void) {
  float stack_data[5 * 10];
  float out_stack_one[10] = {0};
  float out_stack_two[5] = {0};
  float out_stack_three = 0.0f;

  int *heap_data = malloc(5 * 10 * sizeof(int));
  int *out_heap_one = calloc(10, sizeof(int));
  int *out_heap_two = calloc(5, sizeof(int));
  int out_heap_three = 0;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 10; j++) {
      stack_data[i * 10 + j] = (float)(j + 1);
      heap_data[i * 10 + j] = j + 1;
    }
  }

  for (int j = 0; j < 10; j++) {
    float sf = 0.0f;
    int si = 0;
    for (int i = 0; i < 5; i++) {
      sf += stack_data[i * 10 + j];
      si += heap_data[i * 10 + j];
    }
    out_stack_one[j] = sf;
    out_heap_one[j] = si;
  }

  for (int i = 0; i < 5; i++) {
    float sf = 0.0f;
    int si = 0;
    for (int j = 0; j < 10; j++) {
      sf += stack_data[i * 10 + j];
      si += heap_data[i * 10 + j];
    }
    out_stack_two[i] = sf;
    out_heap_two[i] = si;
  }

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 10; j++) {
      out_stack_three += stack_data[i * 10 + j];
      out_heap_three += heap_data[i * 10 + j];
    }
  }

  for (int j = 0; j < 10; j++) {
    if (j < 5) {
      ASSERT(out_stack_two[j] == 55.0f);
      ASSERT(out_heap_two[j] == 55);
    }
    ASSERT(out_stack_one[j] == (float)((j + 1) * 5));
    ASSERT(out_heap_one[j] == (j + 1) * 5);
  }

  ASSERT(out_stack_three == 275.0f);
  ASSERT(out_heap_three == 275);

  free(heap_data);
  free(out_heap_one);
  free(out_heap_two);
}

static void test_product(void) {
  float stack_data[5 * 10];
  float out_stack_one[10];
  float out_stack_two[5];
  float out_stack_three;

  long long *heap_data = malloc(5 * 10 * sizeof(long long));
  long long *out_heap_one = calloc(10, sizeof(long long));
  long long *out_heap_two = calloc(5, sizeof(long long));
  long long out_heap_three;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 10; j++) {
      stack_data[i * 10 + j] = (float)(j + 1);
      heap_data[i * 10 + j] = j + 1;
    }
  }

  for (int j = 0; j < 10; j++) {
    float pf = 1.0f;
    long long pi = 1;
    for (int i = 0; i < 5; i++) {
      pf *= stack_data[i * 10 + j];
      pi *= heap_data[i * 10 + j];
    }
    out_stack_one[j] = pf;
    out_heap_one[j] = pi;
  }

  for (int i = 0; i < 5; i++) {
    float pf = 1.0f;
    long long pi = 1;
    for (int j = 0; j < 10; j++) {
      pf *= stack_data[i * 10 + j];
      pi *= heap_data[i * 10 + j];
    }
    out_stack_two[i] = pf;
    out_heap_two[i] = pi;
  }

  out_stack_three = 1.0f;
  out_heap_three = 1;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 10; j++) {
      out_stack_three *= stack_data[i * 10 + j];
      out_heap_three *= heap_data[i * 10 + j];
    }
  }
  (void)out_stack_three;
  (void)out_heap_three;

  for (int j = 0; j < 10; j++) {
    if (j < 5) {
      ASSERT(out_stack_two[j] == 3.6288e6f);
      ASSERT(out_heap_two[j] == 3628800LL);
    }
    long long expected = 1;
    for (int p = 0; p < 5; p++) expected *= (j + 1);
    ASSERT(out_stack_one[j] == (float)expected);
    ASSERT(out_heap_one[j] == expected);
  }

  free(heap_data);
  free(out_heap_one);
  free(out_heap_two);
}

static void calc(void) {
  test_transpose();
  test_matmul();
  test_sum();
  test_product();
}

int main(void) {
  assert_init(0);
  calc();
  assert_finalize(__FILE__);
  return 0;
}
