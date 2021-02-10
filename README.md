# Luminary

Luminary is a CUDA based Raytracing renderer for Signed Distance Fields.

## Status

A sample containing multiple reflective spheres and multiple point lights with specular reflections is currently implemented which is rendered and saved as a `png` file.

## Goal

- Implementing filters for the `png` routines.
- Implementing primite shapes.
- Implementing some Anti-Aliasing method.
- Implementing some Depth of Field method.
- Implementing custom scene description file format.
- Implementing some motion properties that change the scene between each frame.

The licence for this code can be found in the `LICENCE` file.
The `zlib` library is used for the compression part of the `png` routine. Details about its authors and its licence can be found in `Luminary/lib/zlib/zlib.h`.
