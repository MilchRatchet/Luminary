#ifndef UNITTEST_H
#define UNITTEST_H

#if __cplusplus
extern "C" {
#endif

int unittest_brdf(const float tolerance);
int unittest_random();

#if __cplusplus
}
#endif

#endif /* UNITTEST_H */
