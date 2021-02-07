#ifndef SCENE_H
#define SCENE_H

#include "primitives.h"

struct Camera {
  float x;
  float y;
  float z;
  float dir_x;
  float dir_y;
  float dir_z;
  float fov;  // As the ratio of grid width / 2 to grid distance from camera
} typedef Camera;

struct Scene {
  Camera camera;
  unsigned int far_clip_distance;
  Sphere* spheres;
  unsigned int spheres_length;
  Cuboid* cuboids;
  unsigned int cuboids_length;
  Light* lights;
  unsigned int lights_length;
} typedef Scene;

#endif /* SCENE_H */
