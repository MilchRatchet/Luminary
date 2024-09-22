#include "utils.h"

int main(int argc, char* argv[]) {
  luminary_init();

  LuminaryHost* host;
  LUM_FAILURE_HANDLE(luminary_host_create(&host));

  if (argc > 1) {
    LUM_FAILURE_HANDLE(luminary_host_load_obj_file(host, argv[1]));
  }

  LUM_FAILURE_HANDLE(luminary_host_destroy(&host));

  luminary_shutdown();

  return 0;
}
