#include "instance.h"

void instance_create(Instance** _instance, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(_instance);
  MD_CHECK_NULL_ARGUMENT(host);

  Instance* instance;
  LUM_FAILURE_HANDLE(host_malloc(&instance, sizeof(Instance)));

  LuminaryRendererSettings renderer_settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &renderer_settings));

  display_create(&instance->display, renderer_settings.width, renderer_settings.height);

  *_instance = instance;
}

void instance_run(Instance* instance) {
  bool exit_requested = false;

  while (!exit_requested) {
    display_query_events(instance->display, &exit_requested);
  }
}

void instance_destroy(Instance** instance) {
  MD_CHECK_NULL_ARGUMENT(instance);

  display_destroy(&(*instance)->display);

  LUM_FAILURE_HANDLE(host_free(instance));
}
