#include "utils.h"

int main(int argc, char* argv[]) {
  LuminaryHost* host;

  LUM_FAILURE_HANDLE(luminary_host_create(&host));

  return 0;
}
