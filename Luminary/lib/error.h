#ifndef LUMINARY_ERROR_H
#define LUMINARY_ERROR_H

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>

#define assert(ans, message, _abort) \
  { ___assert((ans), (message), __FILE__, __LINE__, (_abort)); }

inline void ___assert(
  const int value, const char* message, const char* file, const int line, const int abort) {
  if (value == 0) {
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

#endif /* ERROR_H */
