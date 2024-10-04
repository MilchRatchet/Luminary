#include "mandarin_duck.h"

void mandarin_duck_create(MandarinDuck** _duck, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(_duck);
  MD_CHECK_NULL_ARGUMENT(host);

  MandarinDuck* duck;
  LUM_FAILURE_HANDLE(host_malloc(&duck, sizeof(MandarinDuck)));

  LuminaryRendererSettings renderer_settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &renderer_settings));

  display_create(&duck->display, renderer_settings.width, renderer_settings.height);

  *_duck = duck;
}

void mandarin_duck_run(MandarinDuck* duck) {
  bool exit_requested = false;

  while (!exit_requested) {
    display_query_events(duck->display, &exit_requested);
  }
}

void mandarin_duck_destroy(MandarinDuck** duck) {
  MD_CHECK_NULL_ARGUMENT(duck);

  display_destroy(&(*duck)->display);

  LUM_FAILURE_HANDLE(host_free(duck));
}
