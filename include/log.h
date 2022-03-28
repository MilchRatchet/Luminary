#ifndef LOG_H
#define LOG_H

#define log_message(fmt, ...) print_log("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define info_message(fmt, ...) print_info("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define warn_message(fmt, ...) print_warn("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define error_message(fmt, ...) print_error("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define crash_message(fmt, ...) print_crash("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)

#if __cplusplus
extern "C" {
#endif

void init_log(int wl);
void print_log(const char* format, ...);
void print_info(const char* format, ...);
void print_warn(const char* format, ...);
void print_error(const char* format, ...);
void print_crash(const char* format, ...);
void write_log();

#if __cplusplus
}
#endif

#endif /* LOG_H */
