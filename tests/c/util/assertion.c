#include <stdio.h>
#include <stdlib.h>
#include "assertion.h"

static int passed_tests = 0;
static int failed_tests = 0;
static int fail_on_error = 1;

void assert_check(int cond, const char *file, int line) {
  if (!cond) {
    printf("Error in '%s' at line %d\n", file, line);
    if (fail_on_error) {
      exit(-1);
    } else {
      failed_tests++;
    }
  } else {
    passed_tests++;
  }
}

void assert_init(int raise_error_on_fail) {
  fail_on_error = raise_error_on_fail;
}

void assert_finalize(const char *file) {
  if (failed_tests > 0) {
    printf("[FAIL] '%s' Passes: %d Fails: %d\n", file, passed_tests, failed_tests);
  } else {
    printf("[PASS] '%s' Passes: %d\n", file, passed_tests);
  }
}
