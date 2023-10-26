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

`MATERIAL ALPHACUT [FP32]`<br/>
Every alpha value smaller than this value is automatically treated as 0, i.e., fully transparent. Number must be in the range [0,1].

`MATERIAL COLORTRA [INT32]`<br/>
Set 1 to enable colored transparency for geometry, 0 else. Enabling this will use the albedo color to change the color of rays passing through the geometry. Note that this can cause artifacts with some albedo textures.

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

`CAMERA APESHAPE [INT32]`<br/>
Aperture shape of the camera:
 - 0 = Round
 - 1 = Bladed

`CAMERA APEBLACO [INT32]`<br/>
Number of blades used in an aperture with bladed shape. Number must at least 3.

`CAMERA EXPOSURE [FP32]`<br/>
Exposure of the camera. Number must be non-negative.

`CAMERA AUTOEXP_ [INT32]`<br/>
Set 1 to activate auto exposure in realtime mode, 0 else.

`CAMERA MINEXPOS [FP32]`<br/>
Minimum exposure of the camera obtained through auto exposure. Number must be non-negative.

`CAMERA MAXEXPOS [FP32]`<br/>
Maximum exposure of the camera obtained through auto exposure. Number must be non-negative.

`CAMERA BLOOM___ [INT32]`<br/>
Set 1 to activate bloom, 0 else.

`CAMERA BLOOMBLE [FP32]`<br/>
Interpolation value for the bloom. A value of 0 turns bloom off while a value of 1 blurs the whole image. Number must be in the range [0,1].

`CAMERA LENSFLAR [INT32]`<br/>
Set 1 to activate lens flares, 0 else.

`CAMERA LENSFTHR [FP32]`<br/>
Threshold brightness used for the lens flare effect. Number must be non-negative.

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
  - 4 = AgX
  - 5 = AgX Punchy
  - 6 = AgX Custom

`CAMERA AGXSLOPE [FP32]`<br/>
Slope amplification used in the custom AgX tonemapper. Number must be non-negative.

`CAMERA AGXPOWER [FP32]`<br/>
Power applied in the custom AgX tonemapper. Number must be non-negative.

`CAMERA AGXSATUR [FP32]`<br/>
Saturation used in the custom AgX tonemapper. Number must be non-negative.

`CAMERA FILTER__ [INT32]`<br/>
Defines which of the available filters is used:
  - 0 = None
  - 1 = Grayscale
  - 2 = Sepia
  - 3 = Gameboy (4 shades of olive green)
  - 4 = 2 Bit Gray (4 shades of gray)
  - 5 = CRT
  - 6 = Black/White

`CAMERA PURKINJE [INT32]`<br/>
Set 1 to activate purkinje effect.

`CAMERA RUSSIANR [FP32]`<br/>
Factor used in russian roulette. A higher number means rays need to correspond to a smaller throughput to be removed early.

## Sky Settings

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

`SKY MOONTEXO [FP32]`<br/>
Horizontal offset applied to the moon texture.

`SKY SUNSTREN [FP32]`<br/>
Sun light strength. Number should be non-negative.

`SKY OZONEALB [INT32]`<br/>
Set 1 to activate ozone absorption, 0 else.

`SKY STEPS___ [INT32]`<br/>
Number of raymarch steps used in the sky computation. Denser atmospheres require a larger number of steps. For the standard atmosphere settings, 200 steps and higher are recommended for best image quality. Number should be non-negative.

`SKY DENSITY_ [FP32]`<br/>
Density of the atmosphere. Number should be non-negative.

`SKY RAYLEDEN [FP32]`<br/>
Concentration of particles that contribute to Rayleigh scattering. Number should be non-negative.

`SKY MIEDENSI [FP32]`<br/>
Concentration of particles that contribute to Mie scattering. Number should be non-negative.

`SKY OZONEDEN [FP32]`<br/>
Concentration of ozone in atmosphere. Number should be non-negative.

`SKY RAYLEFAL [FP32]`<br/>
Exponential inverse height factor of the concentration distribution of particles that contribute to Rayleigh scattering. Number should be non-negative.

`SKY MIEFALLO [FP32]`<br/>
Exponential inverse height factor of the concentration distribution of insoluble particles that contribute to Mie scattering. Number should be non-negative.

`SKY DIAMETER [FP32]`<br/>
Water droplet diameter. Number must be in the range [0.01,50].

`SKY GROUNDVI [FP32]`<br/>
Ground visibility in kilometers. This acts as a concentration factor to water soluble particles that contribute to Mie scattering. Number should be non-negative.

`SKY OZONETHI [FP32]`<br/>
Thickness of ozone layer. Ozone layer is centered at a height of 25km. Number should be non-negative.

`SKY MSFACTOR [FP32]`<br/>
Factor to the multiscattering contribution. Number should be non-negative.

`SKY STARSEED [INT32]`<br/>
Seed used for the star generation.

`SKY STARINTE [FP32]`<br/>
Light strength of the stars. Number should be non-negative.

`SKY STARNUM_ [INT32]`<br/>
Number of stars generated.

`SKY AERIALPE [INT32]`<br/>
Set 1 to activate aerial perspective, 0 else.

`SKY HDRIACTI [INT32]`<br/>
Set 1 to activate HDRI based sky, 0 else.

`SKY HDRIDIM_ [INT32]`<br/>
Number of pixels of the HDRI sky in each dimension.

`SKY HDRISAMP [INT32]`<br/>
Number of samples used to compute the HDRI sky.

`SKY HDRIMIPB [FP32]`<br/>
Mipmap bias applied to the HDRI sky.

`SKY HDRIORIG [FP32] [FP32] [FP32]`<br/>
Position from which the HDRI sky is computed.

## Cloud Settings

`CLOUD ACTIVE__ [INT32]`<br/>
Set 1 to activate clouds, 0 else.

`CLOUD INSCATTE [INT32]`<br/>
Set 1 to activate atmospheric cloud inscattering in front of the clouds, 0 else.

`CLOUD MIPMAPBI [FP32]`<br/>
Bias applied to the mipmapped sampling of the cloud noise textures. Negative numbers cause higher quality clouds in reflections at the cost of performance. Positive numbers cause lower quality clouds but improves performance.

`CLOUD SEED____ [INT32]`<br/>
Seed used to generate the weather map.

`CLOUD OFFSET__ [FP32] [FP32]`<br/>
Horizontal offset of the clouds.

`CLOUD SHASCALE [FP32]`<br/>
Scaling used when sampling the shape noise texture. Number should be non-negative.

`CLOUD DETSCALE [FP32]`<br/>
Scaling used when sampling the detail noise texture. Number should be non-negative.

`CLOUD WEASCALE [FP32]`<br/>
Scaling used when sampling the weather noise texture. Number should be non-negative.

`CLOUD ANVIL___ [FP32]`<br/>
Anvil overhang. Number must be in the range [0,1].

`CLOUD DIAMETER [FP32]`<br/>
Water droplet diameter. Number must be in the range [0.01,50].

`CLOUD SHASTEPS [INT32]`<br/>
Number of raymarching steps that are used for volumetric extinction/shadowing. Values larger than 10 are generally not necessary. Number should be non-negative.

`CLOUD DENSITY_ [FP32]`<br/>
Cloud density scaling. Number should be non-negative.

`CLOUD LOWACTIV/MIDACTIV/TOPACTIV [INT32]`<br/>
Set 1 to activate low-layer-/mid-layer-/top-layer-clouds, 0 else.

`CLOUD LOWCOVER/MIDCOVER/TOPCOVER [FP32] [FP32]`<br/>
Cloud minimum coverage and coverage factor of the respective cloud layer. Number should be non-negative.

`CLOUD LOWTYPE_/MIDTYPE_/TOPTYPE_ [FP32]`<br/>
Cloud minimum type and type multiplier of the respective cloud layer. Number should be non-negative.

`CLOUD LOWHEIGH/MIDHEIGH/TOPHEIGH [FP32] [FP32]`<br/>
Minimum and maximum height of the respective cloud layer. Number should be non-negative.

`CLOUD LOWWIND_/MIDWIND_/TOPWIND_ [FP32]`<br/>
Wind speed and angle in the respective cloud layer. The angle determines the direction in which the wind blows. Number should be non-negative.

## Fog Settings

`FOG ACTIVE__ [INT32]`<br/>
Set 1 to activate fog, 0 else.

`FOG SCATTERI [FP32]`<br/>
Scattering coefficient determining the amout of light being scattered. Number should be non-negative.

`FOG DIAMETER [FP32]`<br/>
Water droplet diameter. Number must be in the range [0.01,50].

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

`OCEAN REFRACT_ [FP32]`<br/>
Refraction index of the ocean relative to the air. Number must be at least 1.

`OCEAN WATERTYP [INT32]`<br/>
Jerlov water type used for the underwater volume rendering. Higher numbered types are more polluted. The types are as follows:
 - 0 = Open Ocean Type (I)
 - 1 = Open Ocean Type (IA)
 - 2 = Open Ocean Type (IB)
 - 3 = Open Ocean Type (II)
 - 4 = Open Ocean Type (III)
 - 5 = Coastal Type (1C)
 - 6 = Coastal Type (3C)
 - 7 = Coastal Type (5C)
 - 8 = Coastal Type (7C)
 - 9 = Coastal Type (9C)

## Particle Settings

`PARTICLE ACTIVE__ [INT32]`<br/>
Set 1 to activate particles, 0 else.

`PARTICLE SCALE___ [FP32]`<br/>
Scale of the particle field's repeated voxel.

`PARTICLE ALBEDO__ [FP32] [FP32] [FP32]`<br/>
Albedo color of the toy. Numbers must be in the range [0,1].

`PARTICLE DIRECTIO [FP32] [FP32]`<br/>
Direction of the particle motion given by altitude and azimuth.

`PARTICLE SPEED___ [FP32]`<br/>
Magnitude of the particle motion.

`PARTICLE PHASEDIA [FP32]`<br/>
Particle diameter for use in the phase function.

`PARTICLE SEED____ [INT32]`<br/>
Seed used for the generation of the particles.

`PARTICLE COUNT___ [INT32]`<br/>
Number of particles generated within each voxel. The particle field repeats the same generated voxel.

`PARTICLE SIZE____ [FP32]`<br/>
Size of each particle.

`PARTICLE SIZEVARI [FP32]`<br/>
Magnitude of the variation applied to the size of each particle.

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
  - 1 = Plane

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

>📝 A common cause of light leaking / broken lighting is unrealistic vertex normals. Issues appear if the vertex normals represent a surface that is too different from the actual geometry. Performing steps like edge splitting often alleviate this issue at the cost of a higher triangle count.

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
 - Normal Textures (OpenGL format)
   - Red: Tangent X
   - Green: Tangent Y
   - Blue: Tangent Z
   - Alpha: Displacement

>📝 Textures must contain gamma information if they are not encoded linearly. Note that GIMP does not correctly export the correct gamma value. Hence, it is important to uncheck "Save gamma" during png export ([See Thread](https://gitlab.gnome.org/GNOME/gimp/-/issues/5363)). If textures are not displayed correctly, make sure that the gamma value used in the png file correctly approximates the color profile otherwise defined in the file.

Textures are to be referenced in the `*.mtl` as follows:

- map_Kd   = Albedo Textures
- map_Ke   = Illuminance Textures
- map_Ns   = Material Textures
- map_Bump = Normal Textures

In `Blender` these map types correspond to:

- map_Kd   = Albedo
- map_Ke   = Emission
- map_Ns   = Rougness
- map_Bump = Normal

>📝 There was a [bug](https://developer.blender.org/D14519) in Blender before 02.04.22 where map_Ke were not exported. If you encounter any issues of Luminary detecting no emissive triangles, use a newer Blender version for the export.


