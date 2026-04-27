#include "../util/assertion.h"

static void calc(int a, int b) {
  int c, d;

  if (a == 100) {
    c = 23;
    d = 2;
  } else {
    c = 82;
    d = 1;
  }

  if (b == 200 && c == 23) {
    ASSERT(d == 2 && c == 23);
  }

  if (a > 99 && a < 101) {
    ASSERT(1);
  } else {
    ASSERT(0);
  }

  if (a != 100) {
    ASSERT(0);
  }

  if (a != 100 || b != 200) {
    ASSERT(0);
  }

  if (a == 100 || b == 200) {
    ASSERT(1);
  } else {
    ASSERT(0);
  }
}

int main(void) {
  assert_init(0);
  calc(100, 200);
  assert_finalize(__FILE__);
  return 0;
}
