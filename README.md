# Luminary

Luminary is a CUDA based Raytracing renderer for Signed Distance Fields.

## Status

Routines for saving a frame buffer as a `png` are implemented. Working `CUDA` sample is implemented.

## Goal

- Implementing filters for the `png` routines.
- Implementing primite shapes.
- Implementing scene containers, containing shapes.
- Implementing Ray Tracer.
- Implementing some Anti-Aliasing method.
- Implementing some Depth of Field method.
- Implementing custom scene description file format.

The licence for this code can be found in the `LICENCE` file.
The `zlib` library is used for the compression part of the `png` routine. Details about its authors and its licence can be found in `Luminary/lib/zlib/zlib.h`.
