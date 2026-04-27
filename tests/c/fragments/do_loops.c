#include "../util/assertion.h"

static void calc(int a) {
  int n, c;

  c = 0;
  for (n = 1; n <= a; n++) c += n;
  ASSERT(c == 5050);

  c = 0;
  for (n = 1; n <= a; n += 2) c += n;
  ASSERT(c == 2500);

  c = 0;
  for (n = 80; n <= a; n++) c += n;
  ASSERT(c == 1890);

  c = 10000;
  for (n = a; n >= 1; n--) c -= n;
  ASSERT(c == 4950);

  c = 10000;
  for (n = a; n >= 1; n -= 15) c -= n;
  ASSERT(c == 9615);

  c = 10000;
  for (n = a - 80; n >= 1; n--) c -= n;
  ASSERT(c == 9790);

  c = 0;
  for (n = 1; n <= a; n++) {
    c += n;
    if (n > 10) break;
  }
  ASSERT(c == 66);

  c = 0;
  for (n = 1; n <= a; n++) {
    if (n > 20) continue;
    c += n;
  }
  ASSERT(c == 210);
}

int main(void) {
  assert_init(0);
  calc(100);
  assert_finalize(__FILE__);
  return 0;
}
