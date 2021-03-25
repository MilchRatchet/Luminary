#ifndef PROCESSING_H
#define PROCESSING_H

#include "raytrace.h"
#include "image.h"
#include "scene.h"

void frame_buffer_to_8bit_image(Camera camera, raytrace_instance* instance, RGB8* image);
void frame_buffer_to_16bit_image(Camera camera, raytrace_instance* instance, RGB16* image);
void post_bloom(raytrace_instance* instance, const float sigma);
void post_tonemapping(raytrace_instance* instance);

#endif /* PROCESSING_H */
