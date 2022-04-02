#ifndef BENCH_H
#define BENCH_H

#if __cplusplus
extern "C" {
#endif

void bench_activate();
void bench_tic();
void bench_toc(char* text);

#if __cplusplus
}
#endif

#endif /* BENCH_H */
