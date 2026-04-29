#include "../util/assertion.h"

static void calc(int a) {
  int n, c;

  /* Standard do-while: equivalent to a `for (n = 1; n <= a; n++)` summation. */
  c = 0;
  n = 1;
  do {
    c += n;
    n++;
  } while (n <= a);
  ASSERT(c == 5050);

  /* Stride 2. */
  c = 0;
  n = 1;
  do {
    c += n;
    n += 2;
  } while (n <= a);
  ASSERT(c == 2500);

  /* Decreasing counter. */
  c = 10000;
  n = a;
  do {
    c -= n;
    n--;
  } while (n >= 1);
  ASSERT(c == 4950);

  /* Body runs exactly once when the cond is false from the start —
     the distinguishing semantic of do-while vs while. A `while (n < 1)`
     here would never enter the body. */
  c = 0;
  n = 100;
  do {
    c += n;
    n++;
  } while (n < 1);
  ASSERT(c == 100);

  /* Same idea but with a side effect that the loop wouldn't otherwise
     produce: counts that the body executed exactly one time. */
  int hits = 0;
  n = 50;
  do {
    hits++;
  } while (n < 0);
  ASSERT(hits == 1);

  /* `break` inside do-while: bail out before the cond test. */
  c = 0;
  n = 1;
  do {
    c += n;
    if (n > 10) break;
    n++;
  } while (n <= a);
  ASSERT(c == 66);

  /* `continue` inside do-while: skip the rest of the body but still
     evaluate the cond. C semantics — `continue` jumps *to* the test, not
     past it. */
  c = 0;
  n = 0;
  do {
    n++;
    if (n > 20) continue;
    c += n;
  } while (n <= a);
  ASSERT(c == 210);

  /* Nested do-while: outer counts iterations, inner sums 1..3 each time. */
  int outer = 0;
  int inner_sum_total = 0;
  n = 0;
  do {
    int k = 0;
    int s = 0;
    do {
      k++;
      s += k;
    } while (k < 3);
    inner_sum_total += s;
    outer++;
    n++;
  } while (n < 4);
  ASSERT(outer == 4);
  ASSERT(inner_sum_total == 24); /* 4 outer iterations × (1+2+3) = 24 */
}

int main(void) {
  assert_init(0);
  calc(100);
  assert_finalize(__FILE__);
  return 0;
}
