# Scene Description Format

A scene is decribed by a `*.lum` file. An example can be found in `Example.lum`.

## General Settings

`GENERAL WIDTH___ [INT32]`<br/>
The number of horizontal pixels used internally for rendering. Number must be strictly greater than 0.

`GENERAL HEIGHT__ [INT32]`<br/>
The number of vertical pixels used internally for rendering. Number must be strictly greater than 0.

`GENERAL BOUNCES_ [INT32]`<br/>
This number restricts the number of bounces of the path that is traced for each pixel. Number must be non-negative.

`GENERAL SAMPLES_ [INT32]`<br/>
The number of samples to compute per pixel in offline mode. Number must be strictly greater than 0.

`GENERAL DENOISER [INT32]`<br/>
Defines which of the available denoising modes is used:
  - 0 = No denoising
  - 1 = Optix Denoiser
  - 2 = Optix Denoiser with 4x Upscaling

Note that upscaling increases the output image resolution in offline rendering.

`GENERAL OUTPUTFN [STRING]`<br/>
File name of output image in offline mode. Specified directory must already exist.

`GENERAL MESHFILE [STRING]`<br/>
Path to an `*.obj` mesh file. Option may be specified multiple times to load multiple mesh files.

## Material Settings

`MATERIAL LIGHTSON [INT32]`<br/>
Set 1 to enable light sources from geometry, 0 else.

`MATERIAL SMOOTHNE [FP32]`<br/>
Default smoothness value that is used when no material texture is present. Number must be in the range [0,1].

`MATERIAL METALLIC [FP32]`<br/>
Default metallic value that is used when no material texture is present. Number must be in the range [0,1].

`MATERIAL EMISSION [FP32]`<br/>
Default emission intensity value that is used when no material texture is present. Number must be strictly greater than 0.

`MATERIAL FRESNEL_ [INT32]`<br/>
Defines which of the available fresnel approximations is used:
  - 0 = Schlick
  - 1 = Fdez-Aguera

`MATERIAL BVHALPHA [INT32]`<br/>
Set 1 to check for fully transparent hits during BVH traversal, 0 else. This fixes early ray termination in scenes that make much use of alpha cutoff. However, it comes at a mild performance cost.

`MATERIAL ALPHACUT [FP32]`<br/>
Every alpha value smaller than this value is automatically treated as 0, i.e., fully transparent. Number must be in the range [0,1].

## Camera Settings

`CAMERA POSITION [FP32] [FP32] [FP32]`<br/>
Position of the camera.

`CAMERA ROTATION [FP32] [FP32] [FP32]`<br/>
Rotation of the camera.

`CAMERA FOV_____ [FP32]`<br/>
Field of view of the camera. Number must be non-negative.

`CAMERA FOCALLEN [FP32]`<br/>
Focal length of the camera. Number must be non-negative.

`CAMERA APERTURE [FP32]`<br/>
Aperture size of the camera. Number must be non-negative.

`CAMERA EXPOSURE [FP32]`<br/>
Exposure of the camera. Number must be non-negative.

`CAMERA AUTOEXP_ [INT32]`<br/>
Set 1 to activate auto exposure in realtime mode, 0 else.

`CAMERA BLOOM___ [INT32]`<br/>
Set 1 to activate bloom, 0 else.

`CAMERA BLOOMSTR [FP32]`<br/>
Strength of the bloom. Number must be non-negative.

`CAMERA BLOOMTHR [FP32]`<br/>
Threshold of the bloom. Bloom is applied to max(pixel - threshold, 0). Number must be non-negative.

`CAMERA DITHER__ [INT32]`<br/>
Set 1 to activate randomized dithering, 0 else.

`CAMERA FARCLIPD [FP32]`<br/>
Maximum distance a ray may travel. Number must be non-negative.

`CAMERA TONEMAP_ [INT32]`<br/>
Defines which of the available tonemaps is used:
  - 0 = None
  - 1 = ACES
  - 2 = Reinhard
  - 3 = Uncharted 2

`CAMERA FILTER__ [INT32]`<br/>
Defines which of the available filters is used:
  - 0 = None
  - 1 = Grayscale
  - 2 = Sepia
  - 3 = Gameboy (4 shades of olive green)
  - 4 = 2 Bit Gray (4 shades of gray)
  - 5 = CRT
  - 6 = Black/White

## Sky Settings

`SKY SUNCOLOR [FP32] [FP32] [FP32]`<br/>
Base color of the sun. Numbers must be in the range [0,1].

`SKY OFFSET__ [FP32] [FP32] [FP32]`<br/>
Offset of geometry relative to sky. This allows to position the geometry outside the atmosphere.

`SKY AZIMUTH_ [FP32]`<br/>
Azimuth of the sun.

`SKY ALTITUDE [FP32]`<br/>
Altitude of the sun.

`SKY MOONAZIM [FP32]`<br/>
Azimuth of the moon.

`SKY MOONALTI [FP32]`<br/>
Altitude of the moon.

`SKY MOONALBE [FP32]`<br/>
Albedo of the moon. Number must be in the range [0,1].

`SKY INTENSIT [FP32]`<br/>
Intensity of the sky. Number should be non-negative.

`SKY SUNSTREN [FP32]`<br/>
Sun light strength. Number should be non-negative.

`SKY OZONEALB [INT32]`<br/>
Set 1 to activate ozone absorption, 0 else.

`SKY STEPS___ [INT32]`<br/>
Number of raymarch steps used in the sky computation. Number should be non-negative.

`SKY SHASTEPS [INT32]`<br/>
Number of raymarch steps used in the extinction integration in the sky computation. Number should be non-negative.

`SKY DENSITY_ [FP32]`<br/>
Density of the atmosphere. Number should be non-negative.

`SKY STARSEED [INT32]`<br/>
Seed used for the star generation.

`SKY STARINTE [FP32]`<br/>
Light strength of the stars. Number should be non-negative.

`SKY STARNUM_ [INT32]`<br/>
Number of stars generated.

## Cloud Settings

`CLOUD ACTIVE__ [INT32]`<br/>
Set 1 to activate clouds, 0 else.

`CLOUD SEED____ [INT32]`<br/>
Seed used to generate the weather map.

`CLOUD OFFSET__ [FP32] [FP32]`<br/>
Horizontal offset of the clouds.

`CLOUD HEIGHTMA [FP32]`<br/>
Maximum height of the clouds. Number should be non-negative.

`CLOUD HEIGHTMI [FP32]`<br/>
Minimum height of the clouds. Number should be non-negative.

`CLOUD SHASCALE [FP32]`<br/>
Scaling used when sampling the shape noise texture. Number should be non-negative.

`CLOUD DETSCALE [FP32]`<br/>
Scaling used when sampling the detail noise texture. Number should be non-negative.

`CLOUD WEASCALE [FP32]`<br/>
Scaling used when sampling the weather noise texture. Number should be non-negative.

`CLOUD CURSCALE [FP32]`<br/>
Scaling used when sampling the curl noise texture. Number should be non-negative.

`CLOUD COVERAGE [FP32]`<br/>
Cloud coverage, higher numbers imply thicker clouds. Number should be non-negative.

`CLOUD COVERMIN [FP32]`<br/>
Cloud minimum coverage, higher numbers imply denser clouds. Number should be non-negative.

`CLOUD ANVIL___ [FP32]`<br/>
Anvil overhang. Number must be in the range [0,1].

`CLOUD FWDSCATT [FP32]`<br/>
Forward scattering g factor used in mie scattering. Number must be in the range [-1,1].

`CLOUD BWDSCATT [FP32]`<br/>
Backward scattering g factor used in mie scattering. Number must be in the range [-1,1].

`CLOUD SCATLERP [FP32]`<br/>
Interpolation factor between forward and backward scattering in the dual lobe phase function. Number must be in the range [0,1].

`CLOUD WETNESS_ [FP32]`<br/>
Determines how wet the clouds are. Wetter clouds are darker. Number must be in the range [0,1].

`CLOUD POWDER__ [FP32]`<br/>
Determines how strong the beer-powder effect is applied. Number must be in the range [0,1].

`CLOUD SHASTEPS [INT32]`<br/>
Number of raymarching steps that are used for volumetric extinction/shadowing. Number should be non-negative.

`CLOUD DENSITY_ [FP32]`<br/>
Cloud density scaling. Number should be non-negative.

## Fog Settings

`FOG ACTIVE__ [INT32]`<br/>
Set 1 to activate fog, 0 else.

`FOG SCATTERI [FP32]`<br/>
Scattering coefficient determining the amout of light being scattered. Number should be non-negative.

`FOG ANISOTRO [FP32]`<br/>
Anisotropy determines the wether light is more likely to be scattered forward than backward. Numbers must be in the range [0,1].

`FOG DISTANCE [FP32]`<br/>
Maximum distance of the fog from the camera. Number should be non-negative.

`FOG HEIGHT__ [FP32]`<br/>
Ceiling of the fog.

`FOG FALLOFF_ [FP32]`<br/>
Distance over the height of the fog over which the fog density linear goes to 0. Number should be non-negative.

## Ocean Settings

`OCEAN ACTIVE__ [INT32]`<br/>
Set 1 to activate the ocean, 0 else.

`OCEAN HEIGHT__ [FP32]`<br/>
Height of the ocean plane.

`OCEAN AMPLITUD [FP32]`<br/>
Amplitude of the ocean waves. Number should be non-negative.

`OCEAN FREQUENC [FP32]`<br/>
Frequency of the ocean waves. Number should be non-negative.

`OCEAN CHOPPY__ [FP32]`<br/>
Choppyness of the ocean waves. Number should be non-negative.

`OCEAN ANIMATED [INT32]`<br/>
Set 1 to activate the ocean animation, 0 else.

`OCEAN SPEED___ [FP32]`<br/>
Speed of the ocean waves when the ocean is animated. Number should be non-negative.

`OCEAN COLOR___ [FP32] [FP32] [FP32]`<br/>
Albedo of the ocean. Numbers must be in the range [0,1].

`OCEAN EMISSIVE [INT32]`<br/>
Set 1 to make the albedo of the ocean act as emission, 0 else.

`OCEAN REFRACT_ [FP32]`<br/>
Refraction index of the ocean relative to the air. Number must be at least 1.

## Toy Settings

`TOY ACTIVE__ [INT32]`<br/>
Set 1 to activate the toy, 0 else.

`TOY POSITION [FP32] [FP32] [FP32]`<br/>
Position of the toy.

`TOY ROTATION [FP32] [FP32] [FP32]`<br/>
Rotation of the toy.

`TOY SHAPE__ [INT32]`<br/>
Defines which of the available shapes is used for the toy:
  - 0 = Sphere

`TOY SCALE___ [FP32]`<br/>
Scale of the toy.

`TOY COLOR___ [FP32] [FP32] [FP32] [FP32]`<br/>
Albedo color of the toy. The last number represent the alpha. Numbers must be in the range [0,1].

`TOY MATERIAL [FP32] [FP32] [FP32]`<br/>
Material properties of the toy in order: smoothness, metallic and emission strength. First two numbers must be in the range [0,1].

`TOY EMISSIVE [INT32]`<br/>
Set 1 to activate the emission, 0 else.

`TOY EMISSION [FP32] [FP32] [FP32]`<br/>
Emission color of the toy. Numbers must be in the range [0,1].

`TOY REFRACT_ [FP32]`<br/>
Refraction index of the toy relative to the air. Number must be at least 1.

`TOY FLASHLIG [INT32]`<br/>
Set 1 to activate flashlight mode, that is, the toy is behind the camera and emission only happens towards the front, 0 else.

# Meshes

Meshes must be in the Wavefront OBJ (`*.obj`) file format. Geometric vertices, texture coordinates, vertex normals and triangle/quad faces are supported. Textures are to be referenced through a `*.mtl` which has the same name as the `*.obj` file.

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
   - Alpha: Emission Intensity [0,1]
 - Material Textures
   - Red: Smoothness
   - Green: Metallic
   - Blue: Unused
   - Alpha: Unused
 - Normal Textures
   - Red: Tangent X
   - Green: Tangent Y
   - Blue: Tangent Z
   - Alpha: Unused

Textures are to be referenced in the `*.mtl` as follows:

- map_Kd   = Albedo Textures
- map_Ke   = Illuminance Textures
- map_Ns   = Material Textures
- map_Bump = Normal Textures (Spaces are not allowed in path)

In `Blender` these map types correspond to:

- map_Kd   = Albedo
- map_Ke   = Emission
- map_Ns   = Smoothness
- map_Bump = Normal

>📝 There was a [bug](https://developer.blender.org/D14519) in Blender before 02.04.22 where map_Ke were not exported. If you encounter any issues of Luminary detecting no emissive triangles, use a newer Blender version for the export.


