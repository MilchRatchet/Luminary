#include <stdlib.h>
#include <string.h>

#include "argument_parser.h"
#include "mandarin_duck.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  luminary_init();

  ArgumentParser* argument_parser;
  argument_parser_create(&argument_parser);
  argument_parser_parse(argument_parser, argc, (const char**) argv);

  const ArgumentParserResults* results = &argument_parser->results;
  if (results->dry_run_requested || results->invalid_arguments) {
    int status = results->invalid_arguments ? EXIT_FAILURE : EXIT_SUCCESS;

    argument_parser_destroy(&argument_parser);
    luminary_shutdown();
    return status;
  }

  LuminaryHost* host;
  argument_parser_execute(argument_parser, NULL, ARGUMENT_CATEGORY_LUMINARY);
  const LuminaryHostCreateInfo host_info = {.device_mask = results->device_mask};
  LUM_FAILURE_HANDLE(luminary_host_create(&host, host_info));

  argument_parser_execute(argument_parser, host, ARGUMENT_CATEGORY_DEFAULT);
  const MandarinDuckCreateInfo create_info = {
    .mode                  = results->cli_mode ? MANDARIN_DUCK_MODE_CLI : MANDARIN_DUCK_MODE_DEFAULT,
    .host                  = host,
    .initial_window_width  = results->initial_window_width,
    .initial_window_height = results->initial_window_height,
    .output_directory      = results->output_directory,
    .num_benchmark_outputs = results->num_benchmark_outputs,
    .benchmark_name        = results->benchmark_name,
    .cli_samples           = results->cli_samples,
    .cli_format            = results->cli_format,
  };

  argument_parser_destroy(&argument_parser);

  MandarinDuck* duck;
  mandarin_duck_create(&duck, create_info);
  mandarin_duck_run(duck);
  mandarin_duck_destroy(&duck);

  LUM_FAILURE_HANDLE(luminary_host_destroy(&host));
  luminary_shutdown();

  return 0;
}
