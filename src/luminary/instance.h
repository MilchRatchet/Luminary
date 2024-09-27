#ifndef LUMINARY_INSTANCE_H
#define LUMINARY_INSTANCE_H

#include "utils.h"

struct Instance {
  uint32_t mesh_id;
} typedef Instance;

enum InstanceUpdateKind {
  INSTANCE_UPDATE_KIND_UPDATE = 0,
  INSTANCE_UPDATE_KIND_DELETE = 1,
  INSTANCE_UPDATE_KIND_INSERT = 2
} typedef InstanceUpdateKind;

// Have the DeviceManager handle the device updates, then I can do this very easily
// This means we keep a list of instance updates in the scene.
// The device manager takes that list, sends the corresponding updates to the devices,
// and then applies the updates to the actual instance list. This means that the device
// manager is also updating the internal scene. External scene should just always apply
// the update immediatly, so the update list should be empty.
// TODO: Figure out how to handle deletions of instances.
// I could store the instances in banks so that they are easier to move around.

struct InstanceUpdate {
  InstanceUpdateKind kind;
  uint32_t instance_id;
  Instance instance;
} typedef InstanceUpdate;

#endif /* LUMINARY_INSTANCE_H */
