#include "lum_compiler.h"

#include <stdio.h>
#include <string.h>

#include "internal_error.h"
#include "lum_function_tables.h"

#define LUM_VARIABLE_REGISTER_NOT_ASSIGNED 0xFFFFFFFF
#define LUM_VARIABLE_NOT_FOUND 0xFFFFFFFF
#define LUM_COMPILER_MAX_MESSAGE_LENGTH 1024
#define LUM_COMPILER_CONTEXT_STACK_SIZE 16
#define LUM_COMPILER_CONTEXT_STACK_EMPTY 0xFFFFFFFF
#define LUM_FUNCTION_MAX_ARGUMENTS 4

////////////////////////////////////////////////////////////////////
// Compiler Message
////////////////////////////////////////////////////////////////////

enum LumCompilerMessageType {
  LUM_COMPILER_MESSAGE_TYPE_INFO,
  LUM_COMPILER_MESSAGE_TYPE_WARNING,
  LUM_COMPILER_MESSAGE_TYPE_ERROR
} typedef LumCompilerMessageType;

struct LumCompilerMessage {
  LumCompilerMessageType type;
  char message[LUM_COMPILER_MAX_MESSAGE_LENGTH];
  uint32_t line;
  uint32_t col;
} typedef LumCompilerMessage;

static LuminaryResult _lum_compiler_message_init(LumCompilerMessage* message, const LumCompilerMessageType type, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(message);
  __CHECK_NULL_ARGUMENT(token);

  memset(message, 0, sizeof(LumCompilerMessage));

  message->type = type;
  message->line = token->line;
  message->col  = token->col;

  return LUMINARY_SUCCESS;
}

enum LumCompilerContextType {
  LUM_COMPILER_CONTEXT_TYPE_NULL,
  LUM_COMPILER_CONTEXT_TYPE_ACCESS,
  LUM_COMPILER_CONTEXT_TYPE_OPERATOR,
  LUM_COMPILER_CONTEXT_TYPE_INITIALIZER
} typedef LumCompilerContextType;

struct LumAccessContext {
  LumBuiltinType type;
  uint32_t string_stack_object_id;
} typedef LumAccessContext;

struct LumOperatorContext {
  uint32_t dst_stack_object_id;
  uint32_t src_stack_object_id;
} typedef LumOperatorContext;

struct LumCompilerContext {
  LumCompilerContextType type;
  union {
    LumAccessContext access;
    LumOperatorContext operator;
  };
} typedef LumCompilerContext;

////////////////////////////////////////////////////////////////////
// Stack
////////////////////////////////////////////////////////////////////

#define STACK_ALLOCATOR_OBJECT_ID_INVALID (0xFFFFFFFF)

struct LumMemoryObject {
  LumBuiltinType type;
  LumMemoryAllocation allocation;
} typedef LumMemoryObject;

struct LumCompilerStackAllocator {
  uint64_t allocated_stack_size;
  ARRAY LumMemoryObject* allocated_objects;
} typedef LumCompilerStackAllocator;

static LuminaryResult _lum_compiler_stack_allocator_create(LumCompilerStackAllocator** allocator) {
  __CHECK_NULL_ARGUMENT(allocator);

  __FAILURE_HANDLE(host_malloc(allocator, sizeof(LumCompilerStackAllocator)));
  memset(*allocator, 0, sizeof(LumCompilerStackAllocator));

  __FAILURE_HANDLE(array_create(&(*allocator)->allocated_objects, sizeof(LumMemoryObject), 16));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_stack_allocator_reset(LumCompilerStackAllocator* allocator) {
  __CHECK_NULL_ARGUMENT(allocator);

  allocator->allocated_stack_size = 0;

  __FAILURE_HANDLE(array_set_num_elements(allocator->allocated_objects, 0));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_stack_allocator_push(LumCompilerStackAllocator* allocator, LumBuiltinType type, uint32_t* id) {
  __CHECK_NULL_ARGUMENT(allocator);

  const size_t size = lum_builtin_types_sizes[type];

  LumMemoryObject object;
  object.type              = type;
  object.allocation.offset = allocator->allocated_stack_size;
  object.allocation.size   = size;

  allocator->allocated_stack_size += size;

  __FAILURE_HANDLE(array_get_num_elements(allocator->allocated_objects, id));

  __FAILURE_HANDLE(array_push(&allocator->allocated_objects, &object));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_stack_allocator_push_string(LumCompilerStackAllocator* allocator, const char* string, uint32_t* id) {
  __CHECK_NULL_ARGUMENT(allocator);

  const size_t size = strlen(string) + 1;

  LumMemoryObject object;
  object.type              = LUM_BUILTIN_TYPE_STRING;
  object.allocation.offset = allocator->allocated_stack_size;
  object.allocation.size   = size;

  allocator->allocated_stack_size += size;

  __FAILURE_HANDLE(array_get_num_elements(allocator->allocated_objects, id));

  __FAILURE_HANDLE(array_push(&allocator->allocated_objects, &object));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_stack_allocator_destroy(LumCompilerStackAllocator** allocator) {
  __CHECK_NULL_ARGUMENT(allocator);
  __CHECK_NULL_ARGUMENT(*allocator);

  __FAILURE_HANDLE(array_destroy(&(*allocator)->allocated_objects));

  __FAILURE_HANDLE(host_free(allocator));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Compiler State
////////////////////////////////////////////////////////////////////

struct LumCompilerState {
  LumBinary* binary;
  bool error_occurred;
  ARRAY LumCompilerMessage* messages;
  uint32_t returned_stack_object_id;
  uint32_t stack_ptr;
  LumCompilerContext context_stack[LUM_COMPILER_CONTEXT_STACK_SIZE];
  size_t used_stack_size;
  LumCompilerStackAllocator* stack_allocator;
  uint32_t register_variable_map[LUM_REGISTER_COUNT];
  LumTokenizer* tokenizer;
} typedef LumCompilerState;

static LuminaryResult _lum_compiler_state_create(LumCompilerState** state) {
  __CHECK_NULL_ARGUMENT(state);

  __FAILURE_HANDLE(host_malloc(state, sizeof(LumCompilerState)));
  memset(*state, 0, sizeof(LumCompilerState));

  (*state)->stack_ptr = LUM_COMPILER_CONTEXT_STACK_EMPTY;

  __FAILURE_HANDLE(array_create(&(*state)->messages, sizeof(LumCompilerMessage), 16));

  __FAILURE_HANDLE(_lum_compiler_stack_allocator_create(&(*state)->stack_allocator));
  __FAILURE_HANDLE(lum_tokenizer_create(&(*state)->tokenizer));

  memset((*state)->register_variable_map, 0xFF, sizeof(uint32_t) * LUM_REGISTER_COUNT);

  return LUMINARY_SUCCESS;
}

static LumCompilerContextType _lum_compiler_state_get_current_context_type(const LumCompilerState* state) {
  return (state->stack_ptr == LUM_COMPILER_CONTEXT_STACK_EMPTY) ? LUM_COMPILER_CONTEXT_TYPE_NULL
                                                                : state->context_stack[state->stack_ptr].type;
}

static LuminaryResult _lum_compiler_state_destroy(LumCompilerState** state) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(*state);

  __FAILURE_HANDLE(lum_tokenizer_destroy(&(*state)->tokenizer));
  __FAILURE_HANDLE(_lum_compiler_stack_allocator_destroy(&(*state)->stack_allocator));

  __FAILURE_HANDLE(array_destroy(&(*state)->messages));

  __FAILURE_HANDLE(host_free(state));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Instructions
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_emit_return(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_ret(&instruction));

  __FAILURE_HANDLE(array_push(&binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_ldg(LumCompilerState* state, uint32_t dst_stack_id, uint32_t src_stack_id) {
  __CHECK_NULL_ARGUMENT(state);

  LumMemoryObject dst_object = state->stack_allocator->allocated_objects[dst_stack_id];
  LumMemoryObject src_object = state->stack_allocator->allocated_objects[src_stack_id];

  __DEBUG_ASSERT(src_object.type == LUM_BUILTIN_TYPE_STRING);

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_ldg(&instruction, dst_object.type, dst_object.allocation, src_object.allocation));

  __FAILURE_HANDLE(array_push(&state->binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_stg(LumCompilerState* state, uint32_t src_stack_id) {
  __CHECK_NULL_ARGUMENT(state);

  LumMemoryObject src_object = state->stack_allocator->allocated_objects[src_stack_id];

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_stg(&instruction, src_object.type, src_object.allocation));

  __FAILURE_HANDLE(array_push(&state->binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_mov(LumCompilerState* state, uint32_t dst_stack_id, uint32_t src_stack_id) {
  __CHECK_NULL_ARGUMENT(state);

  LumMemoryObject dst_object = state->stack_allocator->allocated_objects[dst_stack_id];
  LumMemoryObject src_object = state->stack_allocator->allocated_objects[src_stack_id];

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_mov(&instruction, dst_object.type, dst_object.allocation, src_object.allocation));

  __FAILURE_HANDLE(array_push(&state->binary->instructions, &instruction));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_null_binary(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  __FAILURE_HANDLE(array_clear(binary->instructions));

  for (uint32_t entrypoint_id = 0; entrypoint_id < LUM_BINARY_ENTRY_POINT_COUNT; entrypoint_id++) {
    binary->entry_points[entrypoint_id] = 0;
  }

  binary->stack_frame_size = 0;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Context resolution
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_context_resolve(LumCompilerState* state) {
  __CHECK_NULL_ARGUMENT(state);

  if (state->stack_ptr == LUM_COMPILER_CONTEXT_STACK_EMPTY) {
    LumCompilerMessage message;
    message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
    sprintf(message.message, "error: internal compiler error, resolved empty stack context");
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
    return LUMINARY_SUCCESS;
  }

  LumCompilerContext* context = state->context_stack + state->stack_ptr;

  switch (context->type) {
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS: {
      if (context->access.type == LUM_BUILTIN_TYPE_VOID) {
        LumCompilerMessage message;
        message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
        sprintf(message.message, "error: expected type identifier before ']'");
        __FAILURE_HANDLE(array_push(&state->messages, &message));

        state->error_occurred = true;
        return LUMINARY_SUCCESS;
      }

      if (context->access.string_stack_object_id == STACK_ALLOCATOR_OBJECT_ID_INVALID) {
        LumCompilerMessage message;
        message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
        sprintf(message.message, "error: expected string literal before ']'");
        __FAILURE_HANDLE(array_push(&state->messages, &message));

        state->error_occurred = true;
        return LUMINARY_SUCCESS;
      }

      uint32_t dst_stack_id;
      __FAILURE_HANDLE(_lum_compiler_stack_allocator_push(state->stack_allocator, context->access.type, &dst_stack_id));

      __FAILURE_HANDLE(_lum_compiler_emit_ldg(state, dst_stack_id, context->access.string_stack_object_id));

      state->returned_stack_object_id = dst_stack_id;
    } break;
    default:
      break;
  }

  state->stack_ptr--;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Identifier
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_identifier_null_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  LumCompilerMessage message;
  message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
  sprintf(message.message, "<source>:%u:%u: error: unexpected identifier", token->line, token->col);
  __FAILURE_HANDLE(array_push(&state->messages, &message));

  state->error_occurred = true;
  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_identifier_access_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  if (token->identifier.is_builtin_type == false) {
    LumCompilerMessage message;
    message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
    sprintf(message.message, "<source>:%u:%u: error: '%s' does not name a type", token->line, token->col, token->identifier.name);
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
    return LUMINARY_SUCCESS;
  }

  if (token->identifier.builtin_type == LUM_BUILTIN_TYPE_VOID) {
    LumCompilerMessage message;
    message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
    sprintf(message.message, "<source>:%u:%u: error: incomplete type 'void' is not allowed", token->line, token->col);
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
    return LUMINARY_SUCCESS;
  }

  state->context_stack[state->stack_ptr].access.type = token->identifier.builtin_type;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_identifier(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL:
      __FAILURE_HANDLE(_lum_compiler_handle_identifier_null_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_handle_identifier_access_context(state, token));
      break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Literal
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_literal_access_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  if (token->literal.type != LUM_LITERAL_TYPE_STRING) {
    LumCompilerMessage message;
    __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_WARNING, token));
    sprintf(message.message, "<source>:%u:%u: error: expected string literal", token->line, token->col);
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    return LUMINARY_SUCCESS;
  }

  uint32_t stack_id;
  __FAILURE_HANDLE(_lum_compiler_stack_allocator_push_string(state->stack_allocator, token->literal.val_string.data, &stack_id));

  state->context_stack[state->stack_ptr].access.string_stack_object_id = stack_id;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_literal(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL: {
      LumCompilerMessage message;
      __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_WARNING, token));
      sprintf(message.message, "<source>:%u:%u: error: unexpected literal", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));
      return LUMINARY_SUCCESS;
    } break;
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_handle_literal_access_context(state, token));
      break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Operator
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_operator(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  if (type != LUM_COMPILER_CONTEXT_TYPE_NULL) {
    LumCompilerMessage message;
    __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_ERROR, token));
    sprintf(message.message, "<source>:%u:%u: error: unexpected operator", token->line, token->col);
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
    return LUMINARY_SUCCESS;
  }

  LumCompilerContext context;
  context.type = LUM_COMPILER_CONTEXT_TYPE_OPERATOR;

  if (state->returned_stack_object_id == STACK_ALLOCATOR_OBJECT_ID_INVALID) {
    LumCompilerMessage message;
    __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_ERROR, token));
    sprintf(message.message, "<source>:%u:%u: error: unexpected operator", token->line, token->col);
    __FAILURE_HANDLE(array_push(&state->messages, &message));

    state->error_occurred = true;
    return LUMINARY_SUCCESS;
  }

  context.operator.dst_stack_object_id = state->returned_stack_object_id;
  context.operator.src_stack_object_id = STACK_ALLOCATOR_OBJECT_ID_INVALID;

  state->context_stack[++state->stack_ptr] = context;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Separator
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_separator_null_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_EOL:
      break;
    case LUM_SEPARATOR_TYPE_ACCESS_BEGIN: {
      LumCompilerContext context;
      context.type                          = LUM_COMPILER_CONTEXT_TYPE_ACCESS;
      context.access.type                   = LUM_BUILTIN_TYPE_VOID;
      context.access.string_stack_object_id = STACK_ALLOCATOR_OBJECT_ID_INVALID;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_END: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of accessor", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected member separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected list separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator_access_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_EOL: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of line", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_BEGIN: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected begin of accessor", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_END: {
      __FAILURE_HANDLE(_lum_compiler_context_resolve(state));
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected member separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected list separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator_operator_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_EOL:
      __FAILURE_HANDLE(_lum_compiler_context_resolve(state));
      break;
    case LUM_SEPARATOR_TYPE_ACCESS_BEGIN: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected begin of accessor", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_END: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected end of accessor", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected member separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      LumCompilerMessage message;
      message.type = LUM_COMPILER_MESSAGE_TYPE_ERROR;
      sprintf(message.message, "<source>:%u:%u: error: unexpected list separator", token->line, token->col);
      __FAILURE_HANDLE(array_push(&state->messages, &message));

      state->error_occurred = true;
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_null_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_OPERATOR:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_operator_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_access_context(state, token));
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////

LuminaryResult lum_compiler_create(LumCompiler** compiler) {
  __CHECK_NULL_ARGUMENT(compiler);

  __FAILURE_HANDLE(host_malloc(compiler, sizeof(LumCompiler)));
  memset(*compiler, 0, sizeof(LumCompiler));

  __FAILURE_HANDLE(_lum_compiler_state_create((LumCompilerState**) &(*compiler)->data));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compiler_compile(LumCompiler* compiler, const LumCompilerCompileInfo* info) {
  __CHECK_NULL_ARGUMENT(compiler);

  LumCompilerState* state = (LumCompilerState*) compiler->data;

  __FAILURE_HANDLE(lum_tokenizer_set_code(state->tokenizer, info->code));

  state->returned_stack_object_id = STACK_ALLOCATOR_OBJECT_ID_INVALID;
  state->binary                   = info->binary;

  LumToken token;

  do {
    __FAILURE_HANDLE(lum_tokenizer_parse_next_token(state->tokenizer, &token));

    if (info->print_parsed_token) {
      __FAILURE_HANDLE(lum_tokenizer_print(state->tokenizer, token));
    }

    switch (token.type) {
      case LUM_TOKEN_TYPE_IDENTIFIER:
        __FAILURE_HANDLE(_lum_compiler_handle_identifier(state, &token));
        break;
      case LUM_TOKEN_TYPE_KEYWORD:
        // No keywords
        break;
      case LUM_TOKEN_TYPE_LITERAL:
        __FAILURE_HANDLE(_lum_compiler_handle_literal(state, &token));
        break;
      case LUM_TOKEN_TYPE_OPERATOR:
        __FAILURE_HANDLE(_lum_compiler_handle_operator(state, &token));
        break;
      case LUM_TOKEN_TYPE_SEPARATOR:
        __FAILURE_HANDLE(_lum_compiler_handle_separator(state, &token));
        break;
      default:
        break;
    }
  } while (token.type != LUM_TOKEN_TYPE_EOF);

  uint32_t message_count;
  __FAILURE_HANDLE(array_get_num_elements(state->messages, &message_count));

  uint32_t num_errors = 0;

  for (uint32_t message_id = 0; message_id < message_count; message_id++) {
    const LumCompilerMessage* message = state->messages + message_id;

    switch (message->type) {
      case LUM_COMPILER_MESSAGE_TYPE_INFO:
        if (compiler->log_level >= 2) {
          info_message("%s", message->message);
        }
        break;
      case LUM_COMPILER_MESSAGE_TYPE_WARNING:
        if (compiler->log_level >= 1) {
          warn_message("%s", message->message);
        }
        break;
      case LUM_COMPILER_MESSAGE_TYPE_ERROR:
        error_message("%s", message->message);
        num_errors++;
        break;
      default:
        break;
    }
  }

  if (num_errors) {
    info_message("%u error%s generated.", num_errors, (num_errors > 1) ? "s" : "");
  }

  __FAILURE_HANDLE(array_clear(state->messages));

  if (state->error_occurred) {
    __FAILURE_HANDLE(_lum_compiler_null_binary(state->binary));
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Failed to compile.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compiler_destroy(LumCompiler** compiler) {
  __CHECK_NULL_ARGUMENT(compiler);
  __CHECK_NULL_ARGUMENT(*compiler);

  __FAILURE_HANDLE(host_free(compiler));

  return LUMINARY_SUCCESS;
}
