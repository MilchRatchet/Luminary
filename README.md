# Luminary

Luminary is a CUDA based Pathtracing renderer.

![Sponza Example](https://github.com/MilchRatchet/Luminary/blob/main/demo_images/Sponza.png)

This project is for fun and to learn more about `Computer Graphics`. Current plans can be found in the `Issues` tab.

As a denoiser I use `Optix` since any non machine learning denoiser is quite frankly not all that great and machine learning is out of the scope of this project.

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
   - Alpha: Unused

Textures are associated to meshes through `*.mtl` files where

- map_Kd = Albedo Textures
- map_Ke = Illuminance Textures
- map_Ns = Material Textures

You can get `Blender` to link the material textures to `map_Ns` by setting them as the input texture for roughness.

A whole scene is arranged through `*.lum` files which are in the following format:
```
Luminary
v 3
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
# Camera lens parameters
# l [Focal Length | Aperture Size | Exposure] (Default Aperture Size=0.0)
l 20.0 0.4 2.0
#
# Sun parameters
# s [Azimuth | Altitude | Intensity]
s 1.0 1.4 50.0
#
# Ocean parameters
# w [Active? | Emissive? | Red | Green | Blue | Alpha | Height | Amplitude | Frequency | Choppyness | Speed]
w 1 0 0.0 0.0 0.0 0.9 222.0 0.5 0.16 4.0 0.8
#
# Rendering parameters
# i [Width | Height | Bounces | Samples per Pixel]
i 1920 1080 6 50
#
# Denoiser (0 = 3x3 Mean, 1 = Optix)
# d [Denoiser] (Default=1)
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

Run Luminary by passing the path of the `*.lum` file.

```
START "" Luminary.exe Scenes/Example.lum
```

Alternatively you can run Luminary in `Realtime Mode` using

```
START "" Luminary.exe Scenes/Example.lum r
```

You can control the camera through `WASD` and the mouse. The sun can be rotated with the arrow keys. You can change parameters by moving the mouse horizontally and pressing the following button:

- `[F]` focal length
- `[G]` aperture size
- `[E]` exposure
- `[C]` alpha cutoff
- `[L]` ocean height
- `[K]` ocean amplitude
- `[J]` ocean frequency
- `[H]` ocean choppyness
- `[Y]` sky density
- `[U]` sky rayleigh falloff
- `[I]` sky mie falloff
- `[N]` default material roughness
- `[M]` default material metallic

Ocean animation can be toggled with `[O]`. Different shading modes can be accessed through `[V]`. The information shown in the title of the window can be switched with `[T]`. Auto Exposure can be toggled with `[R]`. Bloom is toggled by `[B]`. Create an image by pressing `[F12]`. Denoiser can be toggled with `[~]`.

Note that bad performance is to be expected. Path tracing is very computationally expensive, `Luminary` is not very performant yet and `Luminary` does not make use of `RT-Cores` found on `Turing` or `Ampere` architecture graphics cards. The latter would be considered if they would be exposed through `CUDA`.

# Building

This project is a bit of a mess when it comes to building. It was only ever built on `Windows` so changes may have to be made for `Linus/OSX`. Some hints to get it to run are however:

- You need to change the `CUDA toolkit` version in the `CMakeLists.txt` to the one installed on your system.
- You need to change the `CUDA compatibility` version in the `CMakeLists.txt` to your specific version or lower.
- You need to install the `Optix 7.2 SDK` and specify the installation directory in `Luminary/CMake/FindOptix.cmake`.
- You need an `AVX` compatible CPU.
- You need to download the development libraries from http://www.libsdl.org/ and extract the libraries to `Luminary/lib/SDL/`. `SDL2.dll` will automatically be copied to the build directory and has to reside in the same folder as the executable for it to run.

In `Luminary/lib/cuda/directives.cuh` are some preprocessor directives that can be used to tune performance to quality in the CUDA kernel.

# Licences

The licence for this code can be found in the `LICENCE` file.

The `zlib` library is used for the compression part of the `png` routine. Details about its authors and its licence can be found in `Luminary/lib/zlib/zlib.h`.

The `SDL2` library is used for the realtime mode. Details about its authors and its licence can be found in `Luminary/lib/SDL/SDL.h`.

# Literature

This is a list of papers I used for this project so far. Note that some techniques presented in these papers are not implemented at the moment but their ideas were helpful nonetheless:

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
- L. Belcour, D. Coeurjolly, E. Heitz, J. Iehl and V. Ostromoukhov, _A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space_, SIGGRAPH'19 Talks, 2019.
- A. Dietrich, H. Friedrich and M. Stich, _Spatial splits in bounding volume hierarchies_, HPG '09: Proceedings of the Conference on High Performance Graphics 2009, pp. 7-13, 2009.
- A. Ebert, V. Fuetterling, C. Lojewski and F. Pfreundt, _Parallel Spatial Splits in Bounding Volume Hierarchies_, Eurographics Symposium on Parallel Graphics and Visualization, 2016.
- E. Haines, T. Akenine-Möller, "Ray Tracing Gems", Apress, 2019.
