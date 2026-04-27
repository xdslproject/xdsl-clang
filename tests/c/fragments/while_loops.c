#include "../util/assertion.h"

static void calc(int a) {
  int n, c;

  c = 0;
  n = 1;
  while (n <= a) {
    c += n;
    n++;
  }
  ASSERT(c == 5050);

  c = 0;
  n = 1;
  while (n <= a) {
    c += n;
    n += 2;
  }
  ASSERT(c == 2500);

  c = 10000;
  n = a;
  while (n >= 1) {
    c -= n;
    n--;
  }
  ASSERT(c == 4950);

  c = 10000;
  n = a;
  while (n >= 1) {
    c -= n;
    n -= 15;
  }
  ASSERT(c == 9615);

  c = 0;
  n = 1;
  while (n <= a) {
    c += n;
    if (n > 10) break;
    n++;
  }
  ASSERT(c == 66);

  c = 0;
  n = 1;
  while (n <= a) {
    n++;
    if ((n - 1) > 20) continue;
    c += (n - 1);
  }
  ASSERT(c == 210);
}

int main(void) {
  assert_init(0);
  calc(100);
  assert_finalize(__FILE__);
  return 0;
}
