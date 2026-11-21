#include "argument_parser.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

////////////////////////////////////////////////////////////////////
// Argument handler functions
////////////////////////////////////////////////////////////////////

#define UNUSED_ARG(X) (void) X

static void _argument_parser_arg_func_cli(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);
  UNUSED_ARG(arguments);

  parser->results.cli_mode = true;
}

static void _argument_parser_arg_func_format(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);

  const char* format = arguments[0];
  if (strcmp(format, "png") == 0)
    parser->results.cli_format = LUMINARY_IMAGE_FILE_FORMAT_PNG;
  else if (strcmp(format, "jpg") == 0 || strcmp(format, "jpeg") == 0)
    parser->results.cli_format = LUMINARY_IMAGE_FILE_FORMAT_JPG;
  else if (strcmp(format, "bmp") == 0)
    parser->results.cli_format = LUMINARY_IMAGE_FILE_FORMAT_BMP;
  else if (strcmp(format, "tga") == 0)
    parser->results.cli_format = LUMINARY_IMAGE_FILE_FORMAT_TGA;
  else {
    error_message("Unsupported output format '%s'. Expected png, jpg, jpeg, bmp or tga.", format);
    parser->results.invalid_arguments = true;
  }
}

static void _argument_parser_arg_func_help(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);
  UNUSED_ARG(arguments);

  uint32_t num_available_arguments;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->descriptors, &num_available_arguments));

  info_message("OVERVIEW: Mandarin Duck Frontend for Luminary\n\n");

  info_message(
    "USAGE: LuminaryMD [options] file...\n"
    "       LuminaryMD --cli [options] <scene.lum>\n\n"
    "OPTIONS:\n");

  for (uint32_t argument_id = 0; argument_id < num_available_arguments; argument_id++) {
    const ArgumentDescriptor* argument = parser->descriptors + argument_id;

    info_message("\t--%s", argument->long_name);
    if (argument->short_name) {
      info_message(", -%s", argument->short_name);
    }

    const char* mode = argument->mode == ARGUMENT_MODE_CLI_ONLY ? " (CLI only)" : "";
    info_message("\n\t\t%s%s\n", argument->description, mode);
  }

  parser->results.dry_run_requested = true;
}

static void _argument_parser_arg_func_output(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);

  if (num_arguments > 0) {
    parser->results.output_directory = arguments[0];
  }
}

static uint32_t _argument_parser_parse_positive_uint32(const char* value) {
  char* end;
  const char* character = value;

  for (; *character != '\0'; character++) {
    if (*character < '0' || *character > '9')
      return 0;
  }

  errno                              = 0;
  const unsigned long long dimension = strtoull(value, &end, 10);

  if (errno == ERANGE || end == value || *end != '\0' || dimension == 0 || dimension > UINT32_MAX)
    return 0;

  return (uint32_t) dimension;
}

static void _argument_parser_arg_func_samples(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);

  const uint32_t samples = _argument_parser_parse_positive_uint32(arguments[0]);
  if (samples == 0) {
    error_message("Invalid sample count '%s'. Expected a positive 32-bit integer.", arguments[0]);
    parser->results.invalid_arguments = true;
    return;
  }

  parser->results.cli_samples = samples;
}

static void _argument_parser_arg_func_benchmark(
  ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);

  const uint32_t num_outputs = _argument_parser_parse_positive_uint32(arguments[0]);
  if (num_outputs == 0 || num_outputs > 31)
    crash_message("Invalid benchmark exponent: %s. Expected an integer from 1 to 31.", arguments[0]);

  parser->results.num_benchmark_outputs = num_outputs;
  parser->results.benchmark_name        = arguments[1];
}

static uint32_t _argument_parser_parse_dimension(const char* name, const char* value) {
  const uint32_t dimension = _argument_parser_parse_positive_uint32(value);
  if (dimension == 0)
    crash_message("Invalid window %s: %s. Expected a positive 32-bit integer.", name, value);

  return dimension;
}

static void _argument_parser_arg_func_height(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);

  if (num_arguments > 0) {
    parser->results.initial_window_height = _argument_parser_parse_dimension("height", arguments[0]);
  }
}

static void _argument_parser_arg_func_version(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(parser);
  UNUSED_ARG(host);
  UNUSED_ARG(num_arguments);
  UNUSED_ARG(arguments);

  info_message("Mandarin Duck for Luminary %s", MANDARIN_DUCK_VERSION);
  info_message("Build: %s (%s) - %s", MANDARIN_DUCK_BRANCH_NAME, MANDARIN_DUCK_VERSION_HASH, MANDARIN_DUCK_VERSION_DATE);

  parser->results.dry_run_requested = true;
}

static void _argument_parser_arg_func_width(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);

  if (num_arguments > 0) {
    parser->results.initial_window_width = _argument_parser_parse_dimension("width", arguments[0]);
  }
}

static void _argument_parser_arg_func_device(ArgumentParser* parser, LuminaryHost* host, uint32_t num_arguments, const char** arguments) {
  UNUSED_ARG(host);

  if (num_arguments > 0) {
    const uint32_t device_id = atoi(arguments[0]);

    if (device_id > 31)
      crash_message("Invalid device ID %u", device_id);

    // If a device was selected, deselect all devices and only use the onces specified by the user.
    if (parser->results.device_mask == LUMINARY_HOST_CREATE_INFO_DEVICE_MASK_ALL_DEVICES)
      parser->results.device_mask = 0;

    parser->results.device_mask |= 1u << device_id;
  }
}

////////////////////////////////////////////////////////////////////
// Internal functions
////////////////////////////////////////////////////////////////////

struct ParseResult {
  bool is_empty;
  bool is_argument;
  const ArgumentDescriptor* matched_argument;
} typedef ParseResult;

static void _argument_parser_parse_argument(ArgumentParser* parser, const char* argument, ParseResult* result) {
  if (argument[0] == '\0') {
    // This is an empty argument.
    *result = (ParseResult) {.is_empty = true, .is_argument = false, .matched_argument = (const ArgumentDescriptor*) 0};
    return;
  }

  if (argument[0] != '-') {
    // This is an input file
    *result = (ParseResult) {.is_empty = false, .is_argument = false, .matched_argument = (const ArgumentDescriptor*) 0};
    return;
  }

  const bool use_long_name = (argument[1] == '-');

  uint32_t num_available_arguments;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->descriptors, &num_available_arguments));

  const char* arg_string = (use_long_name) ? argument + 2 : argument + 1;

  for (uint32_t argument_id = 0; argument_id < num_available_arguments; argument_id++) {
    const ArgumentDescriptor* argument = parser->descriptors + argument_id;

    // If this argument cannot match, skip it.
    if ((use_long_name && argument->long_name == (char*) 0) || (!use_long_name && argument->short_name == (char*) 0)) {
      continue;
    }

    const int cmp_val = strcmp(arg_string, use_long_name ? argument->long_name : argument->short_name);

    if (cmp_val != 0) {
      continue;
    }

    *result = (ParseResult) {.is_empty = false, .is_argument = true, .matched_argument = argument};

    return;
  }

  *result = (ParseResult) {.is_empty = false, .is_argument = true, .matched_argument = (const ArgumentDescriptor*) 0};
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void argument_parser_create(ArgumentParser** parser) {
  MD_CHECK_NULL_ARGUMENT(parser);

  LUM_FAILURE_HANDLE(host_malloc(parser, sizeof(ArgumentParser)));
  memset(*parser, 0, sizeof(ArgumentParser));

  LUM_FAILURE_HANDLE(array_create(&(*parser)->descriptors, sizeof(ArgumentDescriptor), 16));
  LUM_FAILURE_HANDLE(array_create(&(*parser)->inputs, sizeof(char*), 16));
  LUM_FAILURE_HANDLE(array_create(&(*parser)->parsed_arguments, sizeof(ParsedArgument), 16));

  (*parser)->results.device_mask = LUMINARY_HOST_CREATE_INFO_DEVICE_MASK_ALL_DEVICES;
  (*parser)->results.cli_format  = LUMINARY_IMAGE_FILE_FORMAT_PNG;
  (*parser)->results.cli_samples = 1;

  // Add descriptors sorted by category and alphabetically.

  ArgumentDescriptor descriptor;

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "benchmark";
  descriptor.mode              = ARGUMENT_MODE_CLI_ONLY;
  descriptor.short_name        = "b";
  descriptor.description       = "Save only benchmark checkpoints up to 2^exponent and a timing report: <exponent 1-31> <name>";
  descriptor.subargument_count = 2;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_benchmark;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor = (ArgumentDescriptor) {
    .category     = ARGUMENT_CATEGORY_DEFAULT,
    .mode         = ARGUMENT_MODE_SHARED,
    .long_name    = "cli",
    .description  = "Render without opening a window; save render.<format> unless benchmarking (in --output, if set)",
    .handler_func = _argument_parser_arg_func_cli,
    .pre_execute  = true};
  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor = (ArgumentDescriptor) {
    .category          = ARGUMENT_CATEGORY_DEFAULT,
    .mode              = ARGUMENT_MODE_CLI_ONLY,
    .long_name         = "format",
    .description       = "Set image format: png (default), jpg, jpeg, bmp or tga",
    .subargument_count = 1,
    .handler_func      = _argument_parser_arg_func_format};
  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "help";
  descriptor.mode              = ARGUMENT_MODE_SHARED;
  descriptor.short_name        = "h";
  descriptor.description       = "Print available commandline arguments";
  descriptor.subargument_count = 0;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_help;
  descriptor.pre_execute       = true;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "height";
  descriptor.mode              = ARGUMENT_MODE_SHARED;
  descriptor.short_name        = (char*) 0;
  descriptor.description       = "Set initial window height (render height with --cli) in pixels";
  descriptor.subargument_count = 1;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_height;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "output";
  descriptor.mode              = ARGUMENT_MODE_SHARED;
  descriptor.short_name        = "o";
  descriptor.description       = "Set output directory";
  descriptor.subargument_count = 1;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_output;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "samples";
  descriptor.mode              = ARGUMENT_MODE_CLI_ONLY;
  descriptor.short_name        = "s";
  descriptor.description       = "Set render sample count: <count> (default: 1; ignored with --benchmark)";
  descriptor.subargument_count = 1;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_samples;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "version";
  descriptor.mode              = ARGUMENT_MODE_SHARED;
  descriptor.short_name        = "v";
  descriptor.description       = "Print version information";
  descriptor.subargument_count = 0;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_version;
  descriptor.pre_execute       = true;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_DEFAULT;
  descriptor.long_name         = "width";
  descriptor.mode              = ARGUMENT_MODE_SHARED;
  descriptor.short_name        = (char*) 0;
  descriptor.description       = "Set initial window width (render width with --cli) in pixels";
  descriptor.subargument_count = 1;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_width;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));

  descriptor.category          = ARGUMENT_CATEGORY_LUMINARY;
  descriptor.long_name         = "device";
  descriptor.mode              = ARGUMENT_MODE_SHARED;
  descriptor.short_name        = (char*) 0;
  descriptor.description       = "Specify a device that should be enabled for rendering, specify multiple times for multiple devices";
  descriptor.subargument_count = 1;
  descriptor.handler_func      = (ArgumentHandlerFunc) _argument_parser_arg_func_device;
  descriptor.pre_execute       = false;

  LUM_FAILURE_HANDLE(array_push(&(*parser)->descriptors, &descriptor));
}

void argument_parser_parse(ArgumentParser* parser, uint32_t argc, const char** argv) {
  MD_CHECK_NULL_ARGUMENT(parser);

  uint32_t subargs_left    = 0;
  bool malformed_arguments = false;
  ParseResult subarg_target;
  const char** subargs;
  LUM_FAILURE_HANDLE(array_create(&subargs, sizeof(const char*), 8));

  for (uint32_t argument_id = 1; argument_id < argc; argument_id++) {
    const char* argument = argv[argument_id];

    ParseResult result;
    _argument_parser_parse_argument(parser, argument, &result);

    if (result.is_empty) {
      continue;
    }

    if (result.is_argument) {
      if (subargs_left > 0) {
        warn_message("Missing value for --%s.", subarg_target.matched_argument->long_name);
        malformed_arguments = true;
        if (subarg_target.matched_argument->mode == ARGUMENT_MODE_CLI_ONLY)
          parser->results.invalid_arguments = true;
        subargs_left = 0;
        LUM_FAILURE_HANDLE(array_clear(subargs));
      }
    }

    if (result.is_argument && result.matched_argument == (ArgumentDescriptor*) 0) {
      warn_message("Argument: %s is unknown.", argument);
      malformed_arguments = true;
      continue;
    }

    if (result.is_argument && result.matched_argument->subargument_count == 0) {
      if (result.matched_argument->pre_execute) {
        result.matched_argument->handler_func(parser, NULL, 0, NULL);
        if (parser->results.dry_run_requested)
          break;
        continue;
      }

      const ParsedArgument parsed_argument = {.descriptor = result.matched_argument};

      LUM_FAILURE_HANDLE(array_push(&parser->parsed_arguments, &parsed_argument));

      continue;
    }

    if (result.is_argument && result.matched_argument->subargument_count > 0) {
      subarg_target = result;
      subargs_left  = result.matched_argument->subargument_count;

      continue;
    }

    if (result.is_argument == false && subargs_left > 0) {
      LUM_FAILURE_HANDLE(array_push(&subargs, &argument));
      subargs_left--;

      if (subargs_left == 0) {
        ParsedArgument parsed_argument = {.descriptor = subarg_target.matched_argument};

        memcpy(parsed_argument.subarguments, subargs, subarg_target.matched_argument->subargument_count * sizeof(const char*));

        LUM_FAILURE_HANDLE(array_push(&parser->parsed_arguments, &parsed_argument));

        LUM_FAILURE_HANDLE(array_clear(subargs));
      }

      continue;
    }

    if (!result.is_argument) {
      LUM_FAILURE_HANDLE(array_push(&parser->inputs, &argument));

      continue;
    }
  }

  LUM_FAILURE_HANDLE(array_destroy(&subargs));

  if (parser->results.dry_run_requested)
    return;

  if (subargs_left > 0) {
    warn_message("Missing value for --%s.", subarg_target.matched_argument->long_name);
    malformed_arguments = true;
    if (subarg_target.matched_argument->mode == ARGUMENT_MODE_CLI_ONLY)
      parser->results.invalid_arguments = true;
  }

  uint32_t num_arguments;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->parsed_arguments, &num_arguments));
  for (uint32_t argument_id = 0; argument_id < num_arguments; argument_id++) {
    ParsedArgument* argument             = parser->parsed_arguments + argument_id;
    const ArgumentDescriptor* descriptor = argument->descriptor;
    if (descriptor->mode == ARGUMENT_MODE_CLI_ONLY && parser->results.cli_mode == false) {
      error_message("--%s requires --cli.", descriptor->long_name);
      parser->results.invalid_arguments = true;
    }
    else if (descriptor->mode == ARGUMENT_MODE_CLI_ONLY) {
      descriptor->handler_func(parser, NULL, descriptor->subargument_count, (const char**) argument->subarguments);
    }
  }

  if (parser->results.cli_mode) {
    uint32_t num_inputs;
    LUM_FAILURE_HANDLE(array_get_num_elements(parser->inputs, &num_inputs));
    if (num_inputs != 1) {
      error_message("--cli requires exactly one .lum scene path.");
      parser->results.invalid_arguments = true;
    }
    parser->results.invalid_arguments |= malformed_arguments;
  }
}

void argument_parser_execute(ArgumentParser* parser, LuminaryHost* host, ArgumentCategory category) {
  MD_CHECK_NULL_ARGUMENT(parser);

  if (category == ARGUMENT_CATEGORY_DEFAULT) {
    MD_CHECK_NULL_ARGUMENT(host);

    uint32_t num_inputs;
    LUM_FAILURE_HANDLE(array_get_num_elements(parser->inputs, &num_inputs));

    const uint32_t num_scene_inputs = parser->results.cli_mode ? 1 : num_inputs;
    for (uint32_t input_id = 0; input_id < num_scene_inputs; input_id++) {
      const char* input = parser->inputs[input_id];

      // TODO: Handle obj file inputs
      LuminaryPath* lum_path;
      LUM_FAILURE_HANDLE(luminary_path_create(&lum_path));
      LUM_FAILURE_HANDLE(luminary_path_set_from_string(lum_path, input));

      const LuminaryLoadLumFileArgs load_args = {.path = lum_path, .synchronous = parser->results.cli_mode};
      LUM_FAILURE_HANDLE(luminary_host_load_lum_file(host, &load_args));

      LUM_FAILURE_HANDLE(luminary_path_destroy(&lum_path));
    }
  }

  uint32_t num_arguments;
  LUM_FAILURE_HANDLE(array_get_num_elements(parser->parsed_arguments, &num_arguments));

  for (uint32_t argument_id = 0; argument_id < num_arguments; argument_id++) {
    const ParsedArgument* argument       = parser->parsed_arguments + argument_id;
    const ArgumentDescriptor* descriptor = argument->descriptor;

    if (descriptor->category != category || descriptor->mode == ARGUMENT_MODE_CLI_ONLY)
      continue;

    descriptor->handler_func(parser, host, descriptor->subargument_count, (const char**) argument->subarguments);
  }
}

void argument_parser_destroy(ArgumentParser** parser) {
  MD_CHECK_NULL_ARGUMENT(parser);
  MD_CHECK_NULL_ARGUMENT(*parser);

  LUM_FAILURE_HANDLE(array_destroy(&(*parser)->descriptors));
  LUM_FAILURE_HANDLE(array_destroy(&(*parser)->inputs));
  LUM_FAILURE_HANDLE(array_destroy(&(*parser)->parsed_arguments));

  LUM_FAILURE_HANDLE(host_free(parser));
}
