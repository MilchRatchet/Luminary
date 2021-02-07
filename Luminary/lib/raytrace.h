/*
 * raytrace.h - Routines for raytracing
 */
#include <stdint.h>
#include "scene.h"

#if __cplusplus
extern "C" {
#endif
struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

uint8_t* scene_to_frame(Scene scene, const unsigned int width, const unsigned int height);
#if __cplusplus
}
#endif
