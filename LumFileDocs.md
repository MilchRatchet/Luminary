# Scene Description Format

A scene is decribed by a `*.lum` file which has the following format:
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

# Meshes

Meshes must be in the Wavefront OBJ (`*.obj`) file format. Geometric vertices, texture coordinates, vertex normals and triangle faces are supported. Textures are to be referenced through a `*.mtl` which has the same name as the `*.obj` file.

# Textures

Textures must be in the Portable Network Graphics (`*.png`) file format. They need to have 8 bit channel depth. Supported color formats are `Truecolor` (RGB) and `Truecolor with alpha` (RGBA). They may use filters but may not be interlaced. Textures are used in three different ways:

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

Textures are to be referenced in the `*.mtl` as follows:

- map_Kd = Albedo Textures
- map_Ke = Illuminance Textures
- map_Ns = Material Textures

In `Blender` these map types correspond to:

- map_Kd = Albedo
- map_Ke = Emission
- map_Ns = Smoothness

