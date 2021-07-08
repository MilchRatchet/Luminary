#ifndef PROCESSING_H
#define PROCESSING_H

#include "raytrace.h"
#include "image.h"
#include "scene.h"

void frame_buffer_to_8bit_image(Camera camera, RaytraceInstance* instance, RGB8* image);
void frame_buffer_to_16bit_image(Camera camera, RaytraceInstance* instance, RGB16* image);
void post_bloom(RaytraceInstance* instance, const float sigma, const float strength);
void post_tonemapping(RaytraceInstance* instance);
void post_median_filter(RaytraceInstance*, const float bias);

#endif /* PROCESSING_H */
