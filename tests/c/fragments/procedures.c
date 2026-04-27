#include "../util/assertion.h"

static int v1;
static int v2 = 20;
enum { C_V = 100 };

static void proc_one(int a, int *b) {
  *b = a * 10;
}

static void proc_two(int a, int *b) {
  *b = a * 100;
  return;
  ASSERT(0);
}

static int fn1(int v) { return v * 10; }
static int fn2(int v) { return v * 100; }
static int fn3(int v) { return v * 1000; }
static int fn4(int v) {
  int res = v * 10000;
  return res;
  ASSERT(0);
}

static void mod_globals(void) {
  v1 = 99;
  v2 = 66;
}

static void calc(void) {
  int val;

  ASSERT(v2 == 20);
  ASSERT(C_V == 100);

  v1 = 13;
  ASSERT(v1 == 13);
  v2 = 87;
  ASSERT(v2 == 87);

  mod_globals();
  ASSERT(v1 == 99);
  ASSERT(v2 == 66);

  proc_one(5, &val);
  ASSERT(val == 50);
  proc_two(9, &val);
  ASSERT(val == 900);
  val = fn1(1);
  ASSERT(val == 10);
  val = fn2(2);
  ASSERT(val == 200);
  val = fn3(3);
  ASSERT(val == 3000);
  val = fn4(4);
  ASSERT(val == 40000);
}

int main(void) {
  assert_init(0);
  calc();
  assert_finalize(__FILE__);
  return 0;
}
