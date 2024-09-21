#include "utils.h"

int main(int argc, char* argv[]) {
  luminary_init();

  LuminaryHost* host;

  LUM_FAILURE_HANDLE(luminary_host_create(&host));
  LUM_FAILURE_HANDLE(luminary_host_destroy(&host));

  return 0;
}
