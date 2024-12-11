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

  const bool run_mandarin_duck = !argument_parser->dry_run_requested;

  argument_parser_destroy(&argument_parser);

  if (run_mandarin_duck) {
    MandarinDuck* duck;
    mandarin_duck_create(&duck, host);
    mandarin_duck_run(duck);
    mandarin_duck_destroy(&duck);
  }

  LUM_FAILURE_HANDLE(luminary_host_destroy(&host));

  luminary_shutdown();

  return 0;
}
