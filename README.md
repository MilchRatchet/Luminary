# Luminary

![Daxx Example](/demo_images/Daxx.png)

![Pokitaru Example](/demo_images/Pokitaru.png)

Luminary is a CUDA based pathtracing renderer.

This project is for fun and to learn more about `Computer Graphics`. Current plans can be found in the `Issues` tab.

The goal is to use as few libraries as feasible. Currently, these include `SDL2`, `zlib` and `Optix`. However, only the denoiser is used from the Optix library.

Meshes and textures in the example images are taken from the Ratchet and Clank HD Trilogy and were exported using [Replanetizer](https://github.com/RatchetModding/Replanetizer).

# Usage

The scene is described through the Luminary Scene Description format (`*.lum`). The format is documented in the [Luminary File Documentations](LumFileDocs.md). It is possible to specify a `*.obj` file instead of a `*.lum` file. This will load the mesh and use the default settings. Then one can make changes to the settings and automatically generate a `*.lum` file. Alternatively, one can generate a `*.baked` file that contains all the necessary data in one file. The advantage is that loading from a `*.baked` file is fast, however, the file can be large.

You can start as:

```
Luminary [File] [Option]
```

where `File` is a relative or absolute path to a `*.obj`, `*.lum` or `*.baked` file and Option is one or more of:

```
-o, --offline
        start in offline mode, which renders one image using the specified settings

-t, --timings
        print execution times of some CPU functions

-l, --logs
        write a log file at the end of execution

-s, --samples
        set custom sample count for offline rendering (overrides value set by input file)

-w, --width
        set custom width (overrides value set by input file)

-h, --height
        set custom height (overrides value set by input file)
```

In realtime mode, which is used by default, you can control the camera through `WASD`, `LCTRL`, `SPACE` and the mouse. The sun can be rotated with the arrow keys. A snapshot can be made by pressing `[F12]`. You can open a user interface with `[E]` in which you can change many parameters.

# Building

This project is a bit of a mess when it comes to building. It was only ever built on `Windows` so changes may have to be made for `Linus/OSX`. Some hints to get it to run are:

- You need to change the `CUDA toolkit` version in the `CMakeLists.txt` to the one installed on your system.
- You need to change the `CUDA compatibility` version in the `CMakeLists.txt` to your specific version or lower.
- You need to install the `Optix 7.4 SDK` and specify the installation directory in `Luminary/CMake/FindOptix.cmake`.
- You need an `SSE 4.1` compatible CPU.
- You need to download the development libraries from http://www.libsdl.org/ and https://www.libsdl.org/projects/SDL_ttf/ and extract the libraries to `Luminary/lib/SDL/`. `SDL2.dll`, `SDL2_ttf.dll` and `libfreetype-6.dll` will automatically be copied to the build directory and have to reside in the same folder as the executable for it to run.
- A font file named `LuminaryFont.ttf` must reside in the binary directory. A default font is automatically copied to the build directory. You can replace this font with any other font.
- `_s` versions of `fopen` etc. are used. Those may not be available on `Linus/OSX` and need to be changed there.

In `Luminary/lib/cuda/directives.cuh` are some preprocessor directives that can be used to tune performance to quality in the CUDA kernel.

# Licences

The licence for this code can be found in the `LICENCE` file.

The `zlib` library is used for the compression part of the `png` routine. Details about its authors and its licence can be found in `Luminary/lib/zlib/zlib.h`.

The `SDL2` library is used for the realtime mode. Details about its authors and its licence can be found in `Luminary/lib/SDL/SDL.h`.

The default font provided by `Luminary` is the font `Tuffy` by Ulrich Thatcher which he placed in the `Public Domain`.

# Literature

This is a list of papers I have used for this project so far. Note that some techniques presented in these papers are not implemented at the moment but their ideas were helpful nonetheless:

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
- J. Jimenez, _Next Generation Post Processing in Call of Duty: Advanced Warfare_, Talk at SIGGRAPH 2014.
- A. Marrs, P. Shirley and I. Wald, "Ray Tracing Gems II", Apress, 2021.
- A. Kirk and J. O'Brien, _Perceptually Based Tone Mapping for Low-Light Conditions_, ACM Transactions on Graphics, 30(4), pp. 1-10, 2011.
- J. Patry, _Real-Time Samurai Cinema: Lighting, Atmosphere, and Tonemapping in Ghost of Tsushima_, Talk at SIGGRAPH 2021.
