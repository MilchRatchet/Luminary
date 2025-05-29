#ifndef CU_LUMINARY_SPLITTING_H
#define CU_LUMINARY_SPLITTING_H

#include "utils.cuh"

enum MaterialLayerType : uint8_t {
  MATERIAL_LAYER_TYPE_DIFFUSE,
  MATERIAL_LAYER_TYPE_MICROFACET_REFLECTION,
  MATERIAL_LAYER_TYPE_MICROFACET_REFRACTION
} typedef MaterialLayerType;

struct MaterialLayerInstance {
  MaterialLayerType type;
  uint8_t eval_mask;
} typedef MaterialLayerInstance;

// Increase this when the material system gets expanded
#define MATERIAL_MAX_NUM_LAYERS 2

struct MaterialContext {
  GBufferData data;
  uint32_t num_layers;
  MaterialLayerInstance layer_queue[MATERIAL_MAX_NUM_LAYERS];
} typedef MaterialContext;

#endif /* CU_LUMINARY_SPLITTING_H */
