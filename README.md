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

Requirements:
- CUDA Toolkit 11.6
- Optix 7.4 SDK
- SDL2 and SDL2_ttf
- Modern CMake
- Make or Ninja
- SSE 4.1 compatible CPU
- Supported Nvidia GPU (Recommended: Volta or later)

The `LuminaryFont.ttf` file is automatically copied to the build directory and needs to reside in the same folder as the Luminary executable. You may replace the font with any other font as long as it has the same name. `zlib` comes as a submodule and is compiled with Luminary, it is not required to have `zlib` installed.

## Linux

You need a `nvcc` compatible host compiler. Which compilers are supported can be found in the [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements). In general, any modern GCC, ICC or CLANG will work. By default, `nvcc` uses `gcc`/`g++`.

```
mkdir build
cmake -B ./build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DSDL2_TTF_DIR="{SDL2_TTF Path}" -DOptiX_INSTALL_DIR="{OptiX Path}"
cd build && ninja
cd ..
```
If `cmake` fails to find some packages you will have to specify the directory. For this look at the `Windows` section.

## Windows

You need a modern installation of Visual Studio and a compatible clang-cl and cmake installation (for example as provided in MSYS2). Building was tested on VS 2019, VS 2022 should also work. You need to download SDL2_devel and SDL2_ttf_devel for VC.

You can build using the following commands in the main project directory:
```
mkdir build
call "{VS Path}/VC/Auxiliary/Build/vcvarsall.bat" amd64
cmake -B ./build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="{VS Path}/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" -DCMAKE_C_COMPILER="{MSYS Path}/mingw64/bin/clang-cl.exe" -DSDL2_DIR="{SDL2 Path}" -DSDL2_TTF_DIR="{SDL2_TTF Path}" -DOptiX_INSTALL_DIR="{OptiX Path}"
cd build && ninja
cd ..
```
Notes:
- It is important to use either `clang-cl.exe` or `cl.exe` as the C compiler.
- Run `vcvarsall.bat` only once per terminal.
- The CUDA compiler `nvcc` uses `cl.exe` as host compiler. If `nvcc` failes to find it you need to make sure add its path to the PATH variable. `cl.exe` can be found in `{VS Path}/VC/Tools/MSVC/{Version}/bin/Hostx64/x64`.
- `SDL2.dll` and `SDL2_ttf.dll` are automatically copied into the build dir and always need to reside in the same directory as `Luminary.exe`.

# Licences

The licence for this code can be found in the `LICENCE` file.

The `zlib` library is used for the compression part of the `png` routine. Its licence can be found in the `zlib` repository.

The default font provided by `Luminary` is the font `Tuffy` by Ulrich Thatcher which he placed in the `Public Domain`.

# Literature

This is a list of papers I have used for this project so far. Note that some techniques presented in these papers are not implemented at the moment but their ideas were helpful nonetheless:

- J. Frisvad, _Building an Orthonormal Basis from a 3D Unit Vector Without Normalization_, Journal of Graphics Tools, 16(3), pp. 151-159, 2012
- T. Möller, B. Trumbore, _Fast, Minimum Storage Ray-Triangle Intersection_, Journal of Graphics Tools, 2, pp. 21-28, 1997.
- A. Majercik, C. Crassin, P. Shirley, M. McGuire, _A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering_, Journal of Computer Graphics Techniques, 7(3), pp. 66-82, 2018
- K. Booth, J. MacDonald, _Heuristics for ray tracing using space subdivision_, The Visual Computer, 6, pp. 153-166, 1990.
- T. Karras, S. Laine, H. Ylitie, _Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs_, HPG '17: Proceedings of High Performance Graphics, pp. 1-13, 2017.
- S. Hillaire, _A Scalable and Production Ready Sky and Atmosphere Rendering Technique_, Computer Graphics Forum, 39(4), pp. 13-22, 2020.
- J. Boksansky, _Crash Course in BRDF Implementation_, https://boksajak.github.io/blog/BRDF, 2021.
- S. Lagarde, C. de Rousiers, _Moving Frostbite to Physically Based Rendering_, 2014.
- L. Belcour, D. Coeurjolly, E. Heitz, J. Iehl and V. Ostromoukhov, _A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space_, SIGGRAPH'19 Talks, 2019.
- A. Dietrich, H. Friedrich and M. Stich, _Spatial splits in bounding volume hierarchies_, HPG '09: Proceedings of the Conference on High Performance Graphics 2009, pp. 7-13, 2009.
- E. Haines, T. Akenine-Möller, "Ray Tracing Gems", Apress, 2019.
- J. Jimenez, _Next Generation Post Processing in Call of Duty: Advanced Warfare_, SIGGRAPH 2014.
- A. Marrs, P. Shirley and I. Wald, "Ray Tracing Gems II", Apress, 2021.
- A. Kirk and J. O'Brien, _Perceptually Based Tone Mapping for Low-Light Conditions_, ACM Transactions on Graphics, 30(4), pp. 1-10, 2011.
- J. Patry, _Real-Time Samurai Cinema: Lighting, Atmosphere, and Tonemapping in Ghost of Tsushima_, SIGGRAPH 2021.
- S. Hillaire, _Physically Based Sky, Atmosphere & Cloud Rendering in Frostbite_, SIGGRAPH 2016.
- A. Schneider, _The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn_, SIGGRAPH 2015.
- C. Fdez-Agüera, _A Multiple-Scattering Microfacet Model for Real-Time Image Based Lighting_, Journal of Computer Graphics Techniques (JCGT), 8(1), pp. 45-55, 2019.
- B. Bitterli, C. Wyman, M. Pharr, P. Shirley, A. Lefohn, W. Jarosz, _Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting_, ACM Transactions on Graphics (Proceedings of SIGGRAPH), 39(4), 2020.
