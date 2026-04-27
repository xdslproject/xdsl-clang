#ifndef ASSERTION_H
#define ASSERTION_H

void assert_check(int cond, const char *file, int line);
void assert_init(int raise_error_on_fail);
void assert_finalize(const char *file);

#define ASSERT(cond) assert_check((cond) ? 1 : 0, __FILE__, __LINE__)

#endif
