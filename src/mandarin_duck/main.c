#include "argument_parser.h"
#include "mandarin_duck.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  luminary_init();

  ArgumentParser* argument_parser;
  argument_parser_create(&argument_parser);
  argument_parser_parse(argument_parser, argc, (const char**) argv);

  const bool execute = !argument_parser->results.dry_run_requested;

  LuminaryHost* host;
  MandarinDuckCreateArgs args;

  if (execute) {
    LUM_FAILURE_HANDLE(luminary_host_create(&host));

    argument_parser_execute(argument_parser, host);

    args.mode = (argument_parser->results.num_benchmark_outputs == 0) ? MANDARIN_DUCK_MODE_DEFAULT : MANDARIN_DUCK_MODE_BENCHMARK;
    args.host = host;
    args.output_directory      = argument_parser->results.output_directory;
    args.num_benchmark_outputs = argument_parser->results.num_benchmark_outputs;
  }

  argument_parser_destroy(&argument_parser);

  if (execute) {
    MandarinDuck* duck;
    mandarin_duck_create(&duck, args);
    mandarin_duck_run(duck);
    mandarin_duck_destroy(&duck);

    LUM_FAILURE_HANDLE(luminary_host_destroy(&host));
  }

  luminary_shutdown();

  return 0;
}
