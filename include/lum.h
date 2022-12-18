#ifndef LUM_H
#define LUM_H

#include <stdio.h>

#include "utils.h"
#include "wavefront.h"

int lum_validate_file(FILE* file);
void lum_parse_file(FILE* file, Scene* scene, General* general, WavefrontContent* content);
void lum_write_file(FILE* file, RaytraceInstance* instance);

#endif /* LUM_H */
