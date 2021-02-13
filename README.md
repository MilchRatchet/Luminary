# Luminary

Luminary is a CUDA based Pathtracing renderer for Signed Distance Fields.

## Status

A sample scene containing spheres and cuboids is rendered with simple fully diffuse monte carlo method. Result is stored as a `png`.

## ToDo

- Implement better distribution function for monte carlo method.
- Implement post processing routines (AA, DoF, Bloom).
- Implement filters for the `png` routine.
- Implement custom scene description file format.
- Implement some motion properties that change the scene between each frame.
- Implement refraction. (Maybe)
- Implement volumetric lighting. (Maybe)


The licence for this code can be found in the `LICENCE` file.
The `zlib` library is used for the compression part of the `png` routine. Details about its authors and its licence can be found in `Luminary/lib/zlib/zlib.h`.
