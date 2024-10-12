#include "mandarin_duck.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  luminary_init();

  LuminaryHost* host;
  LUM_FAILURE_HANDLE(luminary_host_create(&host));

  if (argc > 1) {
#if 0
    LuminaryPath* obj_path;
    LUM_FAILURE_HANDLE(luminary_path_create(&obj_path));
    LUM_FAILURE_HANDLE(luminary_path_set_from_string(obj_path, argv[1]));

    LUM_FAILURE_HANDLE(luminary_host_load_obj_file(host, obj_path));

    LUM_FAILURE_HANDLE(luminary_path_destroy(&obj_path));
#else
    LuminaryPath* lum_path;
    LUM_FAILURE_HANDLE(luminary_path_create(&lum_path));
    LUM_FAILURE_HANDLE(luminary_path_set_from_string(lum_path, argv[1]));

    LUM_FAILURE_HANDLE(luminary_host_load_lum_file(host, lum_path));

    LUM_FAILURE_HANDLE(luminary_path_destroy(&lum_path));
#endif
  }

  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  camera.exposure *= 2.0f;

  LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));

  MandarinDuck* duck;
  mandarin_duck_create(&duck, host);
  mandarin_duck_run(duck);
  mandarin_duck_destroy(&duck);

  LUM_FAILURE_HANDLE(luminary_host_destroy(&host));

  luminary_shutdown();

  return 0;
}