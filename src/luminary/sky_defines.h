#ifndef SKY_DEFINES_H
#define SKY_DEFINES_H

#define SKY_EARTH_RADIUS 6371.0f
#define SKY_SUN_RADIUS 696340.0f
#define SKY_SUN_DISTANCE 149597870.0f
#define SKY_MOON_RADIUS 1737.4f
#define SKY_MOON_DISTANCE 384399.0f
#define SKY_ATMO_HEIGHT 100.0f
#define SKY_ATMO_RADIUS (SKY_ATMO_HEIGHT + SKY_EARTH_RADIUS)

#define SKY_MS_TEX_SIZE 32
#define SKY_TM_TEX_WIDTH 256
#define SKY_TM_TEX_HEIGHT 64

// Value must be now larger than (1 << 5) because max Kernel Block Dimension in x is 1024 (for z it is only 64)
#define SKY_MS_BASE (1 << 4)
#define SKY_MS_ITER (SKY_MS_BASE * SKY_MS_BASE)

#endif /* SKY_DEFINES_H */
