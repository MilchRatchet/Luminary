#ifndef PROCESSING_H
#define PROCESSING_H

#include "raytrace.h"
#include "image.h"
#include "scene.h"

void frame_buffer_to_8bit_image(Camera camera, raytrace_instance* instance, RGB8* image);
void frame_buffer_to_16bit_image(Camera camera, raytrace_instance* instance, RGB16* image);

#endif /* PROCESSING_H */
