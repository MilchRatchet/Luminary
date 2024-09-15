#ifndef BENCH_H
#define BENCH_H

#if __cplusplus
extern "C" {
#endif

void bench_activate(void);
void bench_tic(const char* text);
void bench_toc(void);

#if __cplusplus
}
#endif

#endif /* BENCH_H */
