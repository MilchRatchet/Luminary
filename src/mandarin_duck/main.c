#include "argument_parser.h"
#include "mandarin_duck.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  luminary_init();

  LuminaryHost* host;
  LUM_FAILURE_HANDLE(luminary_host_create(&host));

  ArgumentParser* argument_parser;
  argument_parser_create(&argument_parser);
  argument_parser_parse(argument_parser, argc, (const char**) argv, host);
  argument_parser_destroy(&argument_parser);

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
