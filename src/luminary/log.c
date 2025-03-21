#ifdef _WIN32
#include <Windows.h>
#endif

#ifdef _WIN32
#include <wchar.h>

#define ESC "\x1B"
#define CSI "\x1B["
#endif

#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "internal_log.h"
#include "utils.h"

static char* log_buffer;
static size_t log_buffer_size;
static size_t log_buffer_offset;
static char* print_buffer;
static int print_buffer_size;

static mtx_t mutex;

static bool volatile_line = false;

static bool write_logs = true;

static bool log_initialized = false;

#ifdef _WIN32
static void enable_windows_virtual_terminal_sequence() {
  HANDLE hOut  = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD dwMode = 0;
  GetConsoleMode(hOut, &dwMode);
  dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
  SetConsoleMode(hOut, dwMode);
}
#else
#define enable_windows_virtual_terminal_sequence()
#endif /* _WIN32 */

/*
 * Initializes all log functionalities. Using logs before calling this function results in undefined behaviour.
 */
void _log_init(void) {
  log_buffer        = malloc(8192);
  log_buffer_offset = 0;
  log_buffer_size   = 8192;

  print_buffer      = malloc(4096);
  print_buffer_size = 4096;

  enable_windows_virtual_terminal_sequence();

  // This is not allowed to fail.
  const int retval = mtx_init(&mutex, mtx_plain);

  if (retval != thrd_success) {
    puts("Failed to initialize the mutex for the Luminary logger.");
    exit(SIGABRT);
  }

  log_initialized = true;
}

void _log_shutdown(void) {
  luminary_write_log();

  log_initialized = false;

  mtx_destroy(&mutex);

  free(log_buffer);
  free(print_buffer);
}

/*
 * Writes the log to a file.
 */
void luminary_write_log() {
  if (!log_initialized)
    return;

  if (!write_logs)
    return;

  mtx_lock(&mutex);

  FILE* file = fopen("luminary.log", "wb");

  if (!file) {
    mtx_unlock(&mutex);
    luminary_print_error("Could not write log to file.");
    return;
  }

  fwrite(log_buffer, log_buffer_offset, 1, file);
  fclose(file);

  mtx_unlock(&mutex);

  luminary_print_info("Log written to file.");
}

static void exit_program() {
  luminary_write_log();
  puts("Press enter to close...");
  getchar();
  exit(SIGABRT);
}

static void manage_log_buffer(const size_t desired_space) {
  if (log_buffer_size - log_buffer_offset < desired_space) {
    log_buffer_size = log_buffer_offset + desired_space;
    log_buffer      = realloc(log_buffer, log_buffer_size);

    if (!log_buffer)
      exit_program();
  }
}

static void manage_print_buffer(const int desired_space) {
  if (print_buffer_size < desired_space) {
    print_buffer      = realloc(print_buffer, desired_space);
    print_buffer_size = desired_space;

    if (!print_buffer)
      exit_program();
  }
}

static int format_string(const char* format, va_list args) {
  int size = vsnprintf(print_buffer, print_buffer_size, format, args);

  if (size > print_buffer_size) {
    manage_print_buffer(size);
    vsnprintf(print_buffer, print_buffer_size, format, args);
  }

  return size;
}

static void write_to_log_buffer(const size_t size) {
  if (!write_logs)
    return;

  const size_t space = log_buffer_size - log_buffer_offset;

  if (size + 1 > space) {
    manage_log_buffer(size + 1);
  }

  memcpy(log_buffer + log_buffer_offset, print_buffer, size);
  log_buffer_offset += size;

  log_buffer[log_buffer_offset++] = '\n';
}

void luminary_print_log(const char* format, ...) {
  if (!log_initialized)
    return;

  mtx_lock(&mutex);

  va_list args;
  va_start(args, format);
  int size = format_string(format, args);
  va_end(args);
  write_to_log_buffer(size);

  mtx_unlock(&mutex);
}

void luminary_print_info(const char* format, ...) {
  if (!log_initialized)
    return;

  mtx_lock(&mutex);

  va_list args;
  va_start(args, format);
  int size = format_string(format, args);
  va_end(args);
  write_to_log_buffer(size);

  if (volatile_line)
    printf("\33[2K\r");

  printf("\x1B[1m%s\033[0m\n", print_buffer);
  fflush(stdout);

  volatile_line = false;

  mtx_unlock(&mutex);
}

void luminary_print_info_inline(const char* format, ...) {
  if (!log_initialized)
    return;

  mtx_lock(&mutex);

  va_list args;
  va_start(args, format);
  int size = format_string(format, args);
  va_end(args);
  write_to_log_buffer(size);

  if (volatile_line)
    printf("\33[2K\r");

  printf("\x1B[1m%s\033[0m", print_buffer);
  fflush(stdout);

  volatile_line = true;

  mtx_unlock(&mutex);
}

void luminary_print_warn(const char* format, ...) {
  if (!log_initialized)
    return;

  mtx_lock(&mutex);

  va_list args;
  va_start(args, format);
  int size = format_string(format, args);
  va_end(args);
  write_to_log_buffer(size);

  if (volatile_line)
    puts("");

  printf("\x1B[93m\x1B[1m%s\033[0m\n", print_buffer);
  fflush(stdout);

  volatile_line = false;

  mtx_unlock(&mutex);
}

void luminary_print_error(const char* format, ...) {
  if (!log_initialized)
    return;

  mtx_lock(&mutex);

  va_list args;
  va_start(args, format);
  int size = format_string(format, args);
  va_end(args);
  write_to_log_buffer(size);

  if (volatile_line)
    puts("");

  printf("\x1B[91m\x1B[1m%s\033[0m\n", print_buffer);
  fflush(stdout);

  volatile_line = false;

  mtx_unlock(&mutex);
}

void luminary_print_crash(const char* format, ...) {
  if (!log_initialized)
    return;

  mtx_lock(&mutex);

  va_list args;
  va_start(args, format);
  int size = format_string(format, args);
  va_end(args);
  write_to_log_buffer(size);

  if (volatile_line)
    puts("");

  printf("\x1B[95m\x1B[1m%s\033[0m\n", print_buffer);
  fflush(stdout);

  volatile_line = false;

  mtx_unlock(&mutex);

  exit_program();
}
