#include "lib/scene.h"
#include <stdlib.h>

Scene example_scene1() {
  Camera camera = {
    .pos      = {.x = 0.0f, .y = -3.0f, .z = 0.0f},
    .rotation = {.x = 0.3f, .y = -0.7f, .z = 0.0f},
    .fov      = 1.0f};

  Sphere* spheres = (Sphere*) malloc(sizeof(Sphere) * 10);

  spheres[0].id         = 1;
  spheres[0].pos.x      = 0.0f;
  spheres[0].pos.y      = 0.0f;
  spheres[0].pos.z      = -4.0f;
  spheres[0].radius     = 1.0f;
  spheres[0].sign       = 1.0f;
  spheres[0].color.r    = 0.8f;
  spheres[0].color.g    = 0.05f;
  spheres[0].color.b    = 0.05f;
  spheres[0].emission.r = 0.0f;
  spheres[0].emission.g = 0.0f;
  spheres[0].emission.b = 0.0f;
  spheres[0].intensity  = 0.0f;
  spheres[0].smoothness = 1.0f;

  spheres[1].id         = 2;
  spheres[1].pos.x      = 4.0f;
  spheres[1].pos.y      = 0.25f;
  spheres[1].pos.z      = -4.0f;
  spheres[1].radius     = 0.75f;
  spheres[1].sign       = 1.0f;
  spheres[1].color.r    = 0.5f;
  spheres[1].color.g    = 0.5f;
  spheres[1].color.b    = 0.9f;
  spheres[1].emission.r = 0.0f;
  spheres[1].emission.g = 0.0f;
  spheres[1].emission.b = 0.0f;
  spheres[1].intensity  = 0.0f;
  spheres[1].smoothness = 1.0f;

  spheres[2].id         = 3;
  spheres[2].pos.x      = 8.0f;
  spheres[2].pos.y      = 0.25f;
  spheres[2].pos.z      = -5.0f;
  spheres[2].radius     = 0.75f;
  spheres[2].sign       = 1.0f;
  spheres[2].color.r    = 0.9f;
  spheres[2].color.g    = 0.9f;
  spheres[2].color.b    = 0.9f;
  spheres[2].emission.r = 0.0f;
  spheres[2].emission.g = 0.0f;
  spheres[2].emission.b = 0.0f;
  spheres[2].intensity  = 0.0f;
  spheres[2].smoothness = 1.0f;

  spheres[3].id         = 4;
  spheres[3].pos.x      = 12.0f;
  spheres[3].pos.y      = 0.0f;
  spheres[3].pos.z      = -5.0f;
  spheres[3].radius     = 1.0f;
  spheres[3].sign       = 1.0f;
  spheres[3].color.r    = 0.8f;
  spheres[3].color.g    = 0.5f;
  spheres[3].color.b    = 0.1f;
  spheres[3].emission.r = 0.0f;
  spheres[3].emission.g = 0.0f;
  spheres[3].emission.b = 0.0f;
  spheres[3].intensity  = 0.0f;
  spheres[3].smoothness = 1.0f;

  spheres[4].id         = 5;
  spheres[4].pos.x      = 1.0f;
  spheres[4].pos.y      = 0.25f;
  spheres[4].pos.z      = -11.0f;
  spheres[4].radius     = 0.75f;
  spheres[4].sign       = 1.0f;
  spheres[4].color.r    = 0.1f;
  spheres[4].color.g    = 0.1f;
  spheres[4].color.b    = 0.6f;
  spheres[4].emission.r = 0.0f;
  spheres[4].emission.g = 0.0f;
  spheres[4].emission.b = 0.0f;
  spheres[4].intensity  = 0.0f;
  spheres[4].smoothness = 1.0f;

  spheres[5].id         = 6;
  spheres[5].pos.x      = 6.0f;
  spheres[5].pos.y      = -0.5f;
  spheres[5].pos.z      = -12.0f;
  spheres[5].radius     = 1.5f;
  spheres[5].sign       = 1.0f;
  spheres[5].color.r    = 0.9f;
  spheres[5].color.g    = 0.9f;
  spheres[5].color.b    = 0.9f;
  spheres[5].emission.r = 0.0f;
  spheres[5].emission.g = 0.0f;
  spheres[5].emission.b = 0.0f;
  spheres[5].intensity  = 0.0f;
  spheres[5].smoothness = 1.0f;

  spheres[6].id         = 7;
  spheres[6].pos.x      = 0.0f;
  spheres[6].pos.y      = 0.0f;
  spheres[6].pos.z      = -17.0f;
  spheres[6].radius     = 1.0f;
  spheres[6].sign       = 1.0f;
  spheres[6].color.r    = 0.9f;
  spheres[6].color.g    = 0.9f;
  spheres[6].color.b    = 0.9f;
  spheres[6].emission.r = 0.7f;
  spheres[6].emission.g = 0.3f;
  spheres[6].emission.b = 0.1f;
  spheres[6].intensity  = 0.0f;
  spheres[6].smoothness = 1.0f;

  spheres[7].id         = 8;
  spheres[7].pos.x      = 6.0f;
  spheres[7].pos.y      = -0.25f;
  spheres[7].pos.z      = -16.75f;
  spheres[7].radius     = 1.25f;
  spheres[7].sign       = 1.0f;
  spheres[7].color.r    = 0.4f;
  spheres[7].color.g    = 0.4f;
  spheres[7].color.b    = 0.9f;
  spheres[7].emission.r = 0.0f;
  spheres[7].emission.g = 0.0f;
  spheres[7].emission.b = 0.0f;
  spheres[7].intensity  = 0.0f;
  spheres[7].smoothness = 1.0f;

  spheres[8].id         = 9;
  spheres[8].pos.x      = 12.0f;
  spheres[8].pos.y      = 0.5f;
  spheres[8].pos.z      = -17.5f;
  spheres[8].radius     = 0.5f;
  spheres[8].sign       = 1.0f;
  spheres[8].color.r    = 0.9f;
  spheres[8].color.g    = 0.9f;
  spheres[8].color.b    = 0.9f;
  spheres[8].emission.r = 0.0f;
  spheres[8].emission.g = 0.0f;
  spheres[8].emission.b = 0.0f;
  spheres[8].intensity  = 0.0f;
  spheres[8].smoothness = 0.05f;

  spheres[9].id         = 10;
  spheres[9].pos.x      = 12.0f;
  spheres[9].pos.y      = 0.5f;
  spheres[9].pos.z      = -12.0f;
  spheres[9].radius     = 0.5f;
  spheres[9].sign       = 1.0f;
  spheres[9].color.r    = 0.0f;
  spheres[9].color.g    = 0.0f;
  spheres[9].color.b    = 0.0f;
  spheres[9].emission.r = 1.0f;
  spheres[9].emission.g = 1.0f;
  spheres[9].emission.b = 0.9f;
  spheres[9].intensity  = 2.0f;
  spheres[9].smoothness = 0.0f;

  Cuboid* cuboids = (Cuboid*) malloc(sizeof(Cuboid) * 6);

  cuboids[0].id         = 101;
  cuboids[0].pos.x      = 0.0f;
  cuboids[0].pos.y      = 2.0f;
  cuboids[0].pos.z      = 0.0f;
  cuboids[0].size.x     = 20.0f;
  cuboids[0].size.y     = 1.0f;
  cuboids[0].size.z     = 20.0f;
  cuboids[0].sign       = 1.0f;
  cuboids[0].color.r    = 0.95f;
  cuboids[0].color.g    = 0.95f;
  cuboids[0].color.b    = 0.95f;
  cuboids[0].emission.r = 0.0f;
  cuboids[0].emission.g = 0.0f;
  cuboids[0].emission.b = 0.0f;
  cuboids[0].intensity  = 0.0f;
  cuboids[0].smoothness = 0.1f;

  cuboids[1].id         = 102;
  cuboids[1].pos.x      = 14.0f;
  cuboids[1].pos.y      = -2.0f;
  cuboids[1].pos.z      = 0.0f;
  cuboids[1].size.x     = 1.0f;
  cuboids[1].size.y     = 5.0f;
  cuboids[1].size.z     = 20.0f;
  cuboids[1].sign       = 1.0f;
  cuboids[1].color.r    = 0.95f;
  cuboids[1].color.g    = 0.95f;
  cuboids[1].color.b    = 0.95f;
  cuboids[1].emission.r = 0.0f;
  cuboids[1].emission.g = 0.0f;
  cuboids[1].emission.b = 0.0f;
  cuboids[1].intensity  = 0.0f;
  cuboids[1].smoothness = 0.05f;

  cuboids[2].id         = 103;
  cuboids[2].pos.x      = 0.0f;
  cuboids[2].pos.y      = -2.0f;
  cuboids[2].pos.z      = -19.0f;
  cuboids[2].size.x     = 15.0f;
  cuboids[2].size.y     = 5.0f;
  cuboids[2].size.z     = 1.0f;
  cuboids[2].sign       = 1.0f;
  cuboids[2].color.r    = 0.95f;
  cuboids[2].color.g    = 0.95f;
  cuboids[2].color.b    = 0.95f;
  cuboids[2].emission.r = 0.0f;
  cuboids[2].emission.g = 0.0f;
  cuboids[2].emission.b = 0.0f;
  cuboids[2].intensity  = 0.0f;
  cuboids[2].smoothness = 0.05f;

  cuboids[3].id         = 104;
  cuboids[3].pos.x      = 0.0f;
  cuboids[3].pos.y      = -7.0f;
  cuboids[3].pos.z      = 0.0f;
  cuboids[3].size.x     = 20.0f;
  cuboids[3].size.y     = 1.0f;
  cuboids[3].size.z     = 20.0f;
  cuboids[3].sign       = 1.0f;
  cuboids[3].color.r    = 0.95f;
  cuboids[3].color.g    = 0.95f;
  cuboids[3].color.b    = 0.95f;
  cuboids[3].emission.r = 0.0f;
  cuboids[3].emission.g = 0.0f;
  cuboids[3].emission.b = 0.0f;
  cuboids[3].intensity  = 0.0f;
  cuboids[3].smoothness = 0.05f;

  cuboids[4].id         = 105;
  cuboids[4].pos.x      = 14.0f;
  cuboids[4].pos.y      = -2.0f;
  cuboids[4].pos.z      = 0.0f;
  cuboids[4].size.x     = 1.0f;
  cuboids[4].size.y     = 5.0f;
  cuboids[4].size.z     = 20.0f;
  cuboids[4].sign       = 1.0f;
  cuboids[4].color.r    = 0.95f;
  cuboids[4].color.g    = 0.95f;
  cuboids[4].color.b    = 0.95f;
  cuboids[4].emission.r = 0.0f;
  cuboids[4].emission.g = 0.0f;
  cuboids[4].emission.b = 0.0f;
  cuboids[4].intensity  = 0.0f;
  cuboids[4].smoothness = 0.05f;

  cuboids[5].id         = 106;
  cuboids[5].pos.x      = 0.0f;
  cuboids[5].pos.y      = -2.0f;
  cuboids[5].pos.z      = 19.0f;
  cuboids[5].size.x     = 20.0f;
  cuboids[5].size.y     = 5.0f;
  cuboids[5].size.z     = 1.0f;
  cuboids[5].sign       = 1.0f;
  cuboids[5].color.r    = 0.95f;
  cuboids[5].color.g    = 0.95f;
  cuboids[5].color.b    = 0.95f;
  cuboids[5].emission.r = 0.0f;
  cuboids[5].emission.g = 0.0f;
  cuboids[5].emission.b = 0.0f;
  cuboids[5].intensity  = 0.0f;
  cuboids[5].smoothness = 0.05f;

  Scene scene = {
    .camera            = camera,
    .far_clip_distance = 1000,
    .spheres           = spheres,
    .spheres_length    = 10,
    .cuboids           = cuboids,
    .cuboids_length    = 6};

  return scene;
}

Scene example_scene2() {
  Camera camera = {
    .pos      = {.x = 0.0f, .y = 5.0f, .z = 10.0f},
    .rotation = {.x = 0.0f, .y = 0.0f, .z = 0.0f},
    .fov      = 1.0f};

  Sphere* spheres = (Sphere*) malloc(sizeof(Sphere) * 6);

  spheres[0].id         = 1;
  spheres[0].pos.x      = -6.0f;
  spheres[0].pos.y      = 0.0f;
  spheres[0].pos.z      = -10.0f;
  spheres[0].radius     = 1.0f;
  spheres[0].sign       = 1.0f;
  spheres[0].color.r    = 0.7f;
  spheres[0].color.g    = 0.4f;
  spheres[0].color.b    = 0.1f;
  spheres[0].emission.r = 0.0f;
  spheres[0].emission.g = 0.0f;
  spheres[0].emission.b = 0.0f;
  spheres[0].intensity  = 0.0f;
  spheres[0].smoothness = 1.0f;

  spheres[1].id         = 2;
  spheres[1].pos.x      = -3.0f;
  spheres[1].pos.y      = 0.0f;
  spheres[1].pos.z      = -10.0f;
  spheres[1].radius     = 1.0f;
  spheres[1].sign       = 1.0f;
  spheres[1].color.r    = 0.7f;
  spheres[1].color.g    = 0.4f;
  spheres[1].color.b    = 0.1f;
  spheres[1].emission.r = 0.0f;
  spheres[1].emission.g = 0.0f;
  spheres[1].emission.b = 0.0f;
  spheres[1].intensity  = 0.0f;
  spheres[1].smoothness = 0.75f;

  spheres[2].id         = 3;
  spheres[2].pos.x      = 0.0f;
  spheres[2].pos.y      = 0.0f;
  spheres[2].pos.z      = -10.0f;
  spheres[2].radius     = 1.0f;
  spheres[2].sign       = 1.0f;
  spheres[2].color.r    = 0.7f;
  spheres[2].color.g    = 0.4f;
  spheres[2].color.b    = 0.1f;
  spheres[2].emission.r = 0.0f;
  spheres[2].emission.g = 0.0f;
  spheres[2].emission.b = 0.0f;
  spheres[2].intensity  = 0.0f;
  spheres[2].smoothness = 0.5f;

  spheres[3].id         = 4;
  spheres[3].pos.x      = 3.0f;
  spheres[3].pos.y      = 0.0f;
  spheres[3].pos.z      = -10.0f;
  spheres[3].radius     = 1.0f;
  spheres[3].sign       = 1.0f;
  spheres[3].color.r    = 0.7f;
  spheres[3].color.g    = 0.4f;
  spheres[3].color.b    = 0.1f;
  spheres[3].emission.r = 0.0f;
  spheres[3].emission.g = 0.0f;
  spheres[3].emission.b = 0.0f;
  spheres[3].intensity  = 0.0f;
  spheres[3].smoothness = 0.25f;

  spheres[4].id         = 5;
  spheres[4].pos.x      = 6.0f;
  spheres[4].pos.y      = 0.0f;
  spheres[4].pos.z      = -10.0f;
  spheres[4].radius     = 1.0f;
  spheres[4].sign       = 1.0f;
  spheres[4].color.r    = 0.7f;
  spheres[4].color.g    = 0.4f;
  spheres[4].color.b    = 0.1f;
  spheres[4].emission.r = 0.0f;
  spheres[4].emission.g = 0.0f;
  spheres[4].emission.b = 0.0f;
  spheres[4].intensity  = 0.0f;
  spheres[4].smoothness = 0.0f;

  spheres[5].id         = 6;
  spheres[5].pos.x      = 0.0f;
  spheres[5].pos.y      = 7.0f;
  spheres[5].pos.z      = -10.0f;
  spheres[5].radius     = 3.0f;
  spheres[5].sign       = 1.0f;
  spheres[5].color.r    = 1.0f;
  spheres[5].color.g    = 1.0f;
  spheres[5].color.b    = 1.0f;
  spheres[5].emission.r = 0.0f;
  spheres[5].emission.g = 0.0f;
  spheres[5].emission.b = 0.0f;
  spheres[5].intensity  = 0.0f;
  spheres[5].smoothness = 1.0f;

  Cuboid* cuboids = (Cuboid*) malloc(sizeof(Cuboid) * 7);

  cuboids[0].id         = 101;
  cuboids[0].pos.x      = 0.0f;
  cuboids[0].pos.y      = 16.0f;
  cuboids[0].pos.z      = 0.0f;
  cuboids[0].size.x     = 20.0f;
  cuboids[0].size.y     = 1.0f;
  cuboids[0].size.z     = 20.0f;
  cuboids[0].sign       = 1.0f;
  cuboids[0].color.r    = 0.95f;
  cuboids[0].color.g    = 0.95f;
  cuboids[0].color.b    = 0.95f;
  cuboids[0].emission.r = 0.0f;
  cuboids[0].emission.g = 0.0f;
  cuboids[0].emission.b = 0.0f;
  cuboids[0].intensity  = 0.0f;
  cuboids[0].smoothness = 0.1f;

  cuboids[1].id         = 102;
  cuboids[1].pos.x      = 9.0f;
  cuboids[1].pos.y      = 0.0f;
  cuboids[1].pos.z      = 0.0f;
  cuboids[1].size.x     = 1.0f;
  cuboids[1].size.y     = 40.0f;
  cuboids[1].size.z     = 40.0f;
  cuboids[1].sign       = 1.0f;
  cuboids[1].color.r    = 0.1f;
  cuboids[1].color.g    = 1.0f;
  cuboids[1].color.b    = 0.1f;
  cuboids[1].emission.r = 0.0f;
  cuboids[1].emission.g = 0.0f;
  cuboids[1].emission.b = 0.0f;
  cuboids[1].intensity  = 0.0f;
  cuboids[1].smoothness = 0.05f;

  cuboids[2].id         = 103;
  cuboids[2].pos.x      = -9.0f;
  cuboids[2].pos.y      = 0.0f;
  cuboids[2].pos.z      = 0.0f;
  cuboids[2].size.x     = 1.0f;
  cuboids[2].size.y     = 40.0f;
  cuboids[2].size.z     = 40.0f;
  cuboids[2].sign       = 1.0f;
  cuboids[2].color.r    = 1.0f;
  cuboids[2].color.g    = 0.1f;
  cuboids[2].color.b    = 0.1f;
  cuboids[2].emission.r = 0.0f;
  cuboids[2].emission.g = 0.0f;
  cuboids[2].emission.b = 0.0f;
  cuboids[2].intensity  = 0.0f;
  cuboids[2].smoothness = 0.05f;

  cuboids[3].id         = 104;
  cuboids[3].pos.x      = 0.0f;
  cuboids[3].pos.y      = 0.0f;
  cuboids[3].pos.z      = -15.0f;
  cuboids[3].size.x     = 40.0f;
  cuboids[3].size.y     = 40.0f;
  cuboids[3].size.z     = 1.0f;
  cuboids[3].sign       = 1.0f;
  cuboids[3].color.r    = 0.95f;
  cuboids[3].color.g    = 0.95f;
  cuboids[3].color.b    = 0.95f;
  cuboids[3].emission.r = 0.0f;
  cuboids[3].emission.g = 0.0f;
  cuboids[3].emission.b = 0.0f;
  cuboids[3].intensity  = 0.0f;
  cuboids[3].smoothness = 0.05f;

  cuboids[4].id         = 105;
  cuboids[4].pos.x      = 0.0f;
  cuboids[4].pos.y      = 0.0f;
  cuboids[4].pos.z      = 12.0f;
  cuboids[4].size.x     = 40.0f;
  cuboids[4].size.y     = 40.0f;
  cuboids[4].size.z     = 1.0f;
  cuboids[4].sign       = 1.0f;
  cuboids[4].color.r    = 0.95f;
  cuboids[4].color.g    = 0.95f;
  cuboids[4].color.b    = 0.95f;
  cuboids[4].emission.r = 0.0f;
  cuboids[4].emission.g = 0.0f;
  cuboids[4].emission.b = 0.0f;
  cuboids[4].intensity  = 0.0f;
  cuboids[4].smoothness = 0.05f;

  cuboids[5].id         = 106;
  cuboids[5].pos.x      = 0.0f;
  cuboids[5].pos.y      = -5.0f;
  cuboids[5].pos.z      = -10.0f;
  cuboids[5].size.x     = 3.0f;
  cuboids[5].size.y     = 1.0001f;
  cuboids[5].size.z     = 3.0f;
  cuboids[5].sign       = 1.0f;
  cuboids[5].color.r    = 0.95f;
  cuboids[5].color.g    = 0.95f;
  cuboids[5].color.b    = 0.95f;
  cuboids[5].emission.r = 1.0f;
  cuboids[5].emission.g = 1.0f;
  cuboids[5].emission.b = 1.0f;
  cuboids[5].intensity  = 1.0f;
  cuboids[5].smoothness = 0.05f;

  cuboids[6].id         = 107;
  cuboids[6].pos.x      = 0.0f;
  cuboids[6].pos.y      = -5.0f;
  cuboids[6].pos.z      = 0.0f;
  cuboids[6].size.x     = 40.0f;
  cuboids[6].size.y     = 1.0f;
  cuboids[6].size.z     = 40.0f;
  cuboids[6].sign       = 1.0f;
  cuboids[6].color.r    = 0.95f;
  cuboids[6].color.g    = 0.95f;
  cuboids[6].color.b    = 0.95f;
  cuboids[6].emission.r = 0.0f;
  cuboids[6].emission.g = 0.0f;
  cuboids[6].emission.b = 0.0f;
  cuboids[6].intensity  = 0.0f;
  cuboids[6].smoothness = 0.05f;

  Scene scene = {
    .camera            = camera,
    .far_clip_distance = 1000,
    .spheres           = spheres,
    .spheres_length    = 6,
    .cuboids           = cuboids,
    .cuboids_length    = 7};

  return scene;
}
