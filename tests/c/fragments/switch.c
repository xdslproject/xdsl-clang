#include "../util/assertion.h"

// Classify a value with several `case` arms, fall-through between two of
// them, and a `default` arm. The switch operand is a small expression
// (`a + b`) rather than a literal so the lowering exercises a
// non-constant condition.
static int classify(int a, int b) {
  int r = 0;
  switch (a + b) {
    case 1:
      r = 10;
      break;
    case 2:
      r = 20;
      /* fall-through into case 3 */
    case 3:
      r += 100;
      break;
    case 4:
      r = 40;
      break;
    default:
      r = -1;
      break;
  }
  return r;
}

static void calc(void) {
  // 0 + 1 = 1 -> case 1: r = 10
  ASSERT(classify(0, 1) == 10);
  // 1 + 1 = 2 -> case 2 falls into case 3: r = 20 + 100 = 120
  ASSERT(classify(1, 1) == 120);
  // 1 + 2 = 3 -> case 3: r = 0 + 100 = 100
  ASSERT(classify(1, 2) == 100);
  // 2 + 2 = 4 -> case 4: r = 40
  ASSERT(classify(2, 2) == 40);
  // 7 + 0 = 7 -> default: r = -1
  ASSERT(classify(7, 0) == -1);
  // -2 + 1 = -1 -> default: r = -1
  ASSERT(classify(-2, 1) == -1);
}

int main(void) {
  assert_init(0);
  calc();
  assert_finalize(__FILE__);
  return 0;
}
