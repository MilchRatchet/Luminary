#ifndef LUMINARY_ERROR_H
#define LUMINARY_ERROR_H

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>

#define assert(ans, message, _abort) \
  { ___assert((ans), (message), __FILE__, __LINE__, (_abort)); }

inline void ___assert(
  const unsigned long long value, const char* message, const char* file, const int line,
  const int abort) {
  if (!value) {
    fprintf(stderr, "Assertion failed!\nFile: %s\nLine: %d\nMessage: %s\n", file, line, message);
    if (abort) {
      puts("Press enter to close...");
      getchar();
      exit(SIGABRT_COMPAT);
    }
  }
}

#define print_error(message) \
  { ___error((message), __FILE__, __LINE__); }

inline void ___error(const char* message, const char* file, const int line) {
  fprintf(stderr, "Error!\nFile: %s\nLine: %d\nMessage: %s\n", file, line, message);
}

#define safe_realloc(ptr, size) ___s_realloc((ptr), (size));

inline void* ___s_realloc(void* ptr, const size_t size) {
  if (size == 0)
    return (void*) 0;
  void* new_ptr = realloc(ptr, size);
  assert((unsigned long long) new_ptr, "Reallocation failed!", 1);
  return new_ptr;
}

#endif /* ERROR_H */
