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
  MandarinDuckCreateInfo mandarin_duck_create_info;

  if (execute) {
    argument_parser_execute(argument_parser, (LuminaryHost*) 0, ARGUMENT_CATEGORY_LUMINARY);

    LuminaryHostCreateInfo luminary_host_create_info;
    luminary_host_create_info.device_mask = argument_parser->results.device_mask;

    LUM_FAILURE_HANDLE(luminary_host_create(&host, luminary_host_create_info));

    argument_parser_execute(argument_parser, host, ARGUMENT_CATEGORY_DEFAULT);

    mandarin_duck_create_info.mode =
      (argument_parser->results.num_benchmark_outputs == 0) ? MANDARIN_DUCK_MODE_DEFAULT : MANDARIN_DUCK_MODE_BENCHMARK;
    mandarin_duck_create_info.host                  = host;
    mandarin_duck_create_info.output_directory      = argument_parser->results.output_directory;
    mandarin_duck_create_info.num_benchmark_outputs = argument_parser->results.num_benchmark_outputs;
    mandarin_duck_create_info.benchmark_name        = argument_parser->results.benchmark_name;
  }

  argument_parser_destroy(&argument_parser);

  if (execute) {
    MandarinDuck* duck;
    mandarin_duck_create(&duck, mandarin_duck_create_info);
    mandarin_duck_run(duck);
    mandarin_duck_destroy(&duck);

    LUM_FAILURE_HANDLE(luminary_host_destroy(&host));
  }

  luminary_shutdown();

  return 0;
}
