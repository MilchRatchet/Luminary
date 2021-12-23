#include "stars.h"

#include <math.h>
#include <stdlib.h>

#include "bench.h"
#include "buffer.h"
#include "log.h"
#include "utils.h"

static float random() {
  return (float) (((double) rand()) / RAND_MAX);
}

void generate_stars(RaytraceInstance* instance) {
  bench_tic();

  /*
   * Grid [0,2PI] x [-PI/2,PI/2] in blocks of [0,0.1] x [0,0.1]
   * There are 64 x 32 blocks
   */

  const int grid_x = STARS_GRID_LD;
  const int grid_y = 32;

  if (instance->scene_gpu.sky.stars) {
    device_free(instance->scene_gpu.sky.stars, sizeof(Star) * instance->scene_gpu.sky.current_stars_count);
    device_free(instance->scene_gpu.sky.stars_offsets, sizeof(int) * (grid_x * grid_y + 1));
  }

  const int count = instance->scene_gpu.sky.settings_stars_count;
  const int seed  = instance->scene_gpu.sky.stars_seed;

  srand(seed);

  instance->scene_gpu.sky.current_stars_count = count;

  Star* stars = (Star*) malloc(sizeof(Star) * count);

  int* counts = (int*) calloc(grid_x * grid_y, sizeof(int));

  for (int i = 0; i < count; i++) {
    Star s = {
      .altitude  = -PI * 0.5f + PI * (1.0f - sqrtf(random())),
      .azimuth   = 2.0f * PI * random(),
      .radius    = 0.0001f + 0.0014f * (1.0f - sqrtf(random())),
      .intensity = 0.01f * (0.1f + 0.9f * (1.0f - sqrtf(random())))};

    int x = (int) (s.azimuth * 10.0f);
    int y = (int) ((s.altitude + PI * 0.5f) * 10.0f);

    if (x >= grid_x || y >= grid_y || x < 0 || y < 0) {
      error_message("Invalid Grid Position! (%.2f,%.2f) at (%d,%d)", s.azimuth, s.altitude, x, y);
    }

    counts[x + y * grid_x]++;

    stars[i] = s;
  }

  int* offsets = (int*) malloc(sizeof(int) * (grid_x * grid_y + 1));
  Star* grid   = (Star*) malloc(sizeof(Star) * count);

  int offset = 0;
  for (int i = 0; i < grid_x * grid_y; i++) {
    offsets[i] = offset;
    offset += counts[i];
    counts[i] = 0;
  }

  offsets[grid_x * grid_y] = offset;

  for (int i = 0; i < count; i++) {
    Star s = stars[i];

    int x = (int) (s.azimuth * 10.0f);
    int y = (int) ((s.altitude + PI * 0.5f) * 10.0f);

    int p = x + y * grid_x;
    int o = offsets[p];
    int c = counts[p]++;

    grid[o + c] = s;
  }

  free(stars);
  free(counts);

  device_malloc((void**) &instance->scene_gpu.sky.stars, sizeof(Star) * count);
  device_malloc((void**) &instance->scene_gpu.sky.stars_offsets, sizeof(int) * (grid_x * grid_y + 1));

  device_upload(instance->scene_gpu.sky.stars, grid, sizeof(Star) * count);
  device_upload(instance->scene_gpu.sky.stars_offsets, offsets, sizeof(int) * (grid_x * grid_y + 1));

  free(grid);
  free(offsets);

  bench_toc("Generated Stars");
}
