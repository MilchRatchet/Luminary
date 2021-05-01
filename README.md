# Luminary

Luminary is a CUDA based Pathtracing renderer.

![Sponza Example](https://github.com/MilchRatchet/Luminary/blob/main/demo_images/Sponza.png)

This project is for fun and to learn more about Computer Graphics. There is no end goal, I will add whatever I feel like. The following is a list of things that I may do in the future.

- Implement interactive realtime mode.
- Implement refraction.
- Implement better Importance Sampling.
- Implement volumetric lighting.
- Implement post processing routines (AA, DoF).

As a denoiser I use Optix since any non machine learning denoiser is quite frankly not all that great and machine learning is out of the scope of this project.

# Licences

The licence for this code can be found in the `LICENCE` file.

The `zlib` library is used for the compression part of the `png` routine. Details about its authors and its licence can be found in `Luminary/lib/zlib/zlib.h`.

The `SDL2` library is used for the realtime mode. Details about its authors and its licence can be found in `Luminary/lib/SDL/SDL.h`.

# Usage

Meshes need to be in `*.obj` file format. Only triangles are supported. Textures are required to be in 8bit RGBA `png` format. There are three texture types:

 - Albedo Textures
   - Red: Red Color
   - Green: Green Color
   - Blue: Blue Color
   - Alpha: Transparency
 - Illuminance Textures
   - Red: Emission Red Color
   - Green: Emission Green Color
   - Blue: Emission Blue Color
   - Alpha: Unused
 - Material Textures
   - Red: Smoothness
   - Green: Metallic
   - Blue: Emission Intensity

Textures are associated to meshes through `*.mtl` files where

- map_Kd = Albedo Textures
- map_Ke = Illuminance Textures
- map_Ns = Material Textures

A whole scene is arranged through `*.lum` files which are in the following format:
```
Luminary
v 2
# Comments start with a # symbol
# Comments may only appear after the first two lines
# This example demonstrates this particular version of lum
#
# m [Path to Obj file]
m Meshes/Example.obj
#
# Camera parameters
# c [Pos.x | Pos.y | Pos.z | Rotation.x | Rotation.y | Rotation.z | FOV]
c 2.0 0.3 -0.06 0.0 1.570796 0.0 2.0
#
# Sun parameters
# s [Azimuth | Altitude | Intensity]
s 1.0 1.4 50.0
#
# Rendering parameters
# i [Width | Height | Bounces | Samples per Pixel]
i 1920 1080 6 50
#
# BVH depth
# b [Depth]
b 18
# Denoiser (0 = 3x3 Mean, 1 = Optix)
# d [Denoiser]
d 1
#
# Output path (Offline Mode only)
# o [Path to Output file]
o Results/image.png
#
# A *.lum file must contain an x
x
# Lines after the x are ignored
```

Run Luminary by passing the path to the `*.lum` file.

```
START "" Luminary.exe Scenes/Example.lum
```

You can enable a preview window which shows the current progress using

```
START "" Luminary.exe Scenes/Example.lum p
```

Alternatively you can run Luminary in Realtime Mode using

```
START "" Luminary.exe Scenes/Example.lum r
```

# Building

This project is a bit of a mess when it comes to building. Some hints to get it to run are however:

- You require the libraries for SDL to reside in the same folder as the executable (I plan to link them statically in the future once I figure out how)
- You need a modern version of CUDA installed, 10+ should work
- You need to change the CUDA compatibility version in the CMakeLists.txt to your specific version or lower
- You need Optix 7 SDK and you will need to specify the installation directory in `Luminary/CMake/FindOptix.cmake`

# Literature

This is a list of papers I used for this project so far:

- J. Frisvad, _Building an Orthonormal Basis from a 3D Unit Vector Without Normalization_, Journal of Graphics Tools, 16(3), pp. 151-159, 2012
- T. Möller, B. Trumbore, _Fast, Minimum Storage Ray-Triangle Intersection_, Journal of Graphics Tools, 2, pp. 21-28, 1997.
- A. Majercik, C. Crassin, P. Shirley, M. McGuire, _A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering_, Journal of Computer Graphics Techniques, 7(3), pp. 66-82, 2018
- N. Binder, A. Keller, _Efficient Stackless Hierarchy Traversal on GPUs with Backtracking in Constant Time_, HPG '16: Proceedings of High Performance Graphics, pp. 41-50, 2016.
- K. Booth, J. MacDonald, _Heuristics for ray tracing using space subdivision_, The Visual Computer, 6, pp. 153-166, 1990.
- T. Karras, S. Laine, H. Ylitie, _Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs_, HPG '17: Proceedings of High Performance Graphics, pp. 1-13, 2017.
- S. Hillaire, _A Scalable and Production Ready Sky and Atmosphere Rendering Technique_, Computer Graphics Forum, 39(4), pp. 13-22, 2020.
- B. Smolka, M. Szczepanski, K.N. Plataniotis, A.N. Venetsanopoulos, _Fast Modified Vector Median Filter_, Canadian Conference on Electrical and Computer Engineering 2001, 2, pp. 1315-1320, 2001.
- J. Boksansky, _Crash Course in BRDF Implementation_, https://boksajak.github.io/blog/BRDF, 2021.
- S. Lagarde, C. de Rousiers, _Moving Frostbite to Physically Based Rendering_, 2014.
