#ifndef SCENE_H
#define SCENE_H

#include "primitives.h"

/*
 * Camera FOV needs to be done through rotating dir to obtain the grid and giving the rotation limit
 * on the horizontal axis, this avoid fisheye effect
 */

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
  Sphere* spheres;
  unsigned int spheres_length;
  Cuboid* cuboids;
  unsigned int cuboids_length;
} typedef Scene;

#endif /* SCENE_H */
