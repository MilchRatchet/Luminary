# Luminary

Luminary is a CUDA based Pathtracing renderer.

This project is for fun and to learn more about Computer Graphics. There is no end goal, I will add whatever I feel like. The following is a list of things that I may do in the future.

- Implement refraction.
- Implement better distribution function for Monte Carlo method.
- Implement denoiser.
- Implement volumetric lighting.
- Implement post processing routines (AA, DoF).
- Implement realtime rendering mode.

The licence for this code can be found in the `LICENCE` file.
The `zlib` library is used for the compression part of the `png` routine. Details about its authors and its licence can be found in `Luminary/lib/zlib/zlib.h`.

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
