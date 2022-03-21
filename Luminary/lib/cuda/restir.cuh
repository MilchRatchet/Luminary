#ifndef CU_RESTIR_H
#define CU_RESTIR_H

#include "brdf.cuh"
#include "utils.cuh"

/*
 * There is one big issue, we will need the normal to properly weight
 * the samples, however, the normal is not available before shading and
 * we cannot perform restir during ReSTIR. There are some ideas, like computing the normals
 * in the generation phase which is probably somewhat expensive.
 *
 * Maybe it will be possible to unify the shading into a single kernel which is fed by the
 * individual kernels that do the shading right now. This is possible because
 * Geometry, Ocean and Toy all use the same shading anyway and the others do not need
 * ReSTIR.
 */

/*
 * To solve this we do the following
 *  - Buffer 1 contains for each pixel position and normal (24 bytes)
 *  - Buffer 2 and 3 contain the samples
 * Buffer 1 gets created in the process of creating the shading tasks
 * Normal is only needed for geometry and toy
 * For toy we can simply compute it
 * For geometry we store the face_normal in the padding of the traversal triangles,
 * all we then have to do is copy that face normal and reorient it to face against the incoming ray
 *
 * Alternatively we could store not the normal but some data like hit type and reorient hint
 * This would save 8 bytes per pixel but we would need to recompute the normal at every resample
 * It is unclear what is the better option.
 */
__global__ void restir_generate_samples() {
  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < device_amount; offset += blockDim.x * gridDim.x) {
    const RestirEvalData data = load_restir_eval_data(offset);
    RestirSample sample;
    if (data.flags) {
      sample = sample_light(data.position);
    }
    else {
      sample.id     = LIGHT_ID_NONE;
      sample.weight = 0.0f;
    }
    store_restir_sample(device.restir_samples, sample, offset);

    if (device_iteration_type != TYPE_CAMERA) {
      device.restir_eval_data[offset].flags = 0;
    }
  }
}

__global__ void restir_spatial_resampling(RestirSample* input, RestirSample* output) {
  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < device_amount; offset += blockDim.x * gridDim.x) {
    const RestirEvalData data  = load_restir_eval_data(offset);
    const RestirSample current = load_restir_sample(input, offset);

    RestirSample selected = current;

    if (data.flags) {
      const int x = offset % device_width;
      const int y = offset / device_width;

      for (int i = 0; i < device_restir_spatial_samples; i++) {
        int sample_x = x + (int) (2.0f * (white_noise() - 0.5f) * 30.0f);
        int sample_y = y + (int) (2.0f * (white_noise() - 0.5f) * 30.0f);

        sample_x = max(sample_x, 0);
        sample_y = max(sample_y, 0);
        sample_x = min(sample_x, device_width - 1);
        sample_y = min(sample_y, device_height - 1);

        const RestirSample spatial = load_restir_sample(input, sample_x + sample_y * device_width);

        if (spatial.id == LIGHT_ID_NONE)
          continue;

        selected = resample_light(current, spatial, data);
      }
    }

    store_restir_sample(output, selected, offset);
  }
}

/*
 * There is a kernel which performs the sampling for all pixels before the shading kernels.
 * This is done regardless of what the actual hit was so that each pixel has the same number of neighbours.
 * (This makes no sense since position is relevant. While fog hits may be ok, a sky hit has no meaningful
 * position and hence no meaningful way of weighting samples, setting the sample to some default is a must here)
 *
 * We need to keep track of the hit anyway since we don't need resampling on pixels with sky hit.
 *
 * Spatial neighbours on the border are clamped to itself. This is probably not correct but it will
 * only affect the border pixels and improves performance all around has we require less branching etc.
 *
 * We first check whether a neighbour has the same ID, if it does we don't need to do anything.
 * Else we need to reevaluate the weight for our position.
 * Depending on what the final performance will be like it may be good to have an option to turn this
 * reevaluation off. If I understand correctly the biased ReSTIR does not reevaluate either.
 *
 * The weight of the temporal sample is set to 0 if the visibility fails.
 *
 * If both the current and temporal sample are the same but the temporal one has a
 * weight of 0 we cannot just stop tracing a light because it could be only
 * partially obscured. This is not very nice but it has to be done.
 *
 * For the spatial iterations we have an extra kernel and we abuse the temporal buffer
 * for the memory space as we swap buffers in each iteration.
 * This is very important as we are already very limited on memory.
 *
 * We could temporally reproject samples.
 */

#endif /* CU_RESTIR_H */
