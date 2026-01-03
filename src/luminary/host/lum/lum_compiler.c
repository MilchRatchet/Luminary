#include "lum_compiler.h"

#include <stdarg.h>
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
  LUM_COMPILER_CONTEXT_TYPE_MEMBER_ACCESS,
  LUM_COMPILER_CONTEXT_TYPE_OPERATOR,
  LUM_COMPILER_CONTEXT_TYPE_INITIALIZER
} typedef LumCompilerContextType;

struct LumAccessContext {
  LumBuiltinType type;
  uint32_t string_constant_object_id;
} typedef LumAccessContext;

struct LumMemberAccessContext {
  uint32_t base_stack_object_id;
  uint32_t member_stack_object_id;
} typedef LumMemberAccessContext;

struct LumOperatorContext {
  uint32_t dst_stack_object_id;
  uint32_t src_object_id;
} typedef LumOperatorContext;

struct LumInitializerContext {
  uint32_t stack_object_id;
} typedef LumInitializerContext;

struct LumCompilerContext {
  LumCompilerContextType type;
  union {
    LumAccessContext access;
    LumMemberAccessContext member_access;
    LumOperatorContext operator;
    LumInitializerContext initializer;
  };
} typedef LumCompilerContext;

////////////////////////////////////////////////////////////////////
// Constant Memory
////////////////////////////////////////////////////////////////////

#define ALLOCATOR_OBJECT_ID_INVALID (0xFFFFFFFF)
#define ALLOCATOR_OBJECT_ID_CONSTANT (0x80000000)
#define ALLOCATOR_OBJECT_ID_MASK (0x7FFFFFFF)

struct LumMemoryObject {
  LumBuiltinType type;
  LumMemoryAllocation allocation;
} typedef LumMemoryObject;

struct LumConstantMemoryObject {
  LumMemoryObject obj;
  void* data;
} typedef LumConstantMemoryObject;

struct LumCompilerConstantAllocator {
  ARRAY LumConstantMemoryObject* allocated_objects;
  size_t allocated_bytes;
} typedef LumCompilerConstantAllocator;

static LuminaryResult _lum_compiler_constant_allocator_create(LumCompilerConstantAllocator** allocator) {
  __CHECK_NULL_ARGUMENT(allocator);

  __FAILURE_HANDLE(host_malloc(allocator, sizeof(LumCompilerConstantAllocator)));
  memset(*allocator, 0, sizeof(LumCompilerConstantAllocator));

  __FAILURE_HANDLE(array_create(&(*allocator)->allocated_objects, sizeof(LumConstantMemoryObject), 16));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_constant_allocator_reset(LumCompilerConstantAllocator* allocator) {
  __CHECK_NULL_ARGUMENT(allocator);

  uint32_t num_objects;
  __FAILURE_HANDLE(array_get_num_elements(allocator->allocated_objects, &num_objects));

  for (uint32_t obj_id = 0; obj_id < num_objects; obj_id++) {
    __FAILURE_HANDLE(host_free(&allocator->allocated_objects[obj_id].data));
  }

  __FAILURE_HANDLE(array_clear(allocator->allocated_objects));

  allocator->allocated_bytes = 0;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_constant_allocator_push_literal(
  LumCompilerConstantAllocator* allocator, const LumTokenLiteral* literal, uint32_t* id) {
  __CHECK_NULL_ARGUMENT(allocator);
  __CHECK_NULL_ARGUMENT(literal);

  uint32_t num_objects;
  __FAILURE_HANDLE(array_get_num_elements(allocator->allocated_objects, &num_objects));

  const LumBuiltinType type = lum_tokenizer_literal_type_to_builtin[literal->type];
  const size_t size         = (type != LUM_BUILTIN_TYPE_STRING) ? lum_builtin_types_sizes[type] : literal->val_string.length;

  const void* src;
  switch (literal->type) {
    case LUM_LITERAL_TYPE_FLOAT:
      src = (const void*) &literal->val_float;
      break;
    case LUM_LITERAL_TYPE_UINT:
      src = (const void*) &literal->val_uint;
      break;
    case LUM_LITERAL_TYPE_BOOL:
      src = (const void*) &literal->val_bool;
      break;
    case LUM_LITERAL_TYPE_ENUM:
      src = (const void*) &literal->val_enum;
      break;
    case LUM_LITERAL_TYPE_STRING:
      src = (const void*) literal->val_string.data;
      break;
    default:
      __DEBUG_ASSERT(false);
      break;
  }

  bool found_match = false;

  for (uint32_t obj_id = 0; obj_id < num_objects; obj_id++) {
    if (allocator->allocated_objects[obj_id].obj.type == type) {
      if (memcmp(allocator->allocated_objects[obj_id].data, src, size) == 0) {
        *id         = obj_id | ALLOCATOR_OBJECT_ID_CONSTANT;
        found_match = true;
        break;
      }
    }
  }

  if (found_match)
    return LUMINARY_SUCCESS;

  LumConstantMemoryObject obj;
  obj.obj.type              = type;
  obj.obj.allocation.offset = allocator->allocated_bytes | LUM_MEMORY_CONSTANT_MEMORY_SPACE_BIT;
  obj.obj.allocation.size   = size;

  __FAILURE_HANDLE(host_malloc(&obj.data, size));
  memcpy(obj.data, src, size);

  allocator->allocated_bytes += size;

  *id = num_objects | ALLOCATOR_OBJECT_ID_CONSTANT;

  __FAILURE_HANDLE(array_push(&allocator->allocated_objects, &obj));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_constant_allocator_get_memory_object(
  LumCompilerConstantAllocator* allocator, uint32_t id, LumMemoryObject* mem_obj) {
  __CHECK_NULL_ARGUMENT(allocator);
  __CHECK_NULL_ARGUMENT(mem_obj);

  __DEBUG_ASSERT(id != ALLOCATOR_OBJECT_ID_INVALID);
  __DEBUG_ASSERT((id & ALLOCATOR_OBJECT_ID_CONSTANT) != 0);

  *mem_obj = allocator->allocated_objects[id & ALLOCATOR_OBJECT_ID_MASK].obj;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_constant_allocator_create_data_section(LumCompilerConstantAllocator* allocator, LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(allocator);
  __CHECK_NULL_ARGUMENT(binary);

  uint8_t* constant_memory;
  __FAILURE_HANDLE(host_malloc(&constant_memory, allocator->allocated_bytes));

  uint32_t num_objects;
  __FAILURE_HANDLE(array_get_num_elements(allocator->allocated_objects, &num_objects));

  for (uint32_t obj_id = 0; obj_id < num_objects; obj_id++) {
    const LumConstantMemoryObject* obj = allocator->allocated_objects + obj_id;

    memcpy(constant_memory + (obj->obj.allocation.offset & LUM_MEMORY_OFFSET_MASK), obj->data, obj->obj.allocation.size);
  }

  binary->constant_memory      = (void*) constant_memory;
  binary->constant_memory_size = allocator->allocated_bytes;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_constant_allocator_destroy(LumCompilerConstantAllocator** allocator) {
  __CHECK_NULL_ARGUMENT(allocator);
  __CHECK_NULL_ARGUMENT(*allocator);

  __FAILURE_HANDLE(_lum_compiler_constant_allocator_reset(*allocator));

  __FAILURE_HANDLE(array_destroy(&(*allocator)->allocated_objects));

  __FAILURE_HANDLE(host_free(allocator));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Stack
////////////////////////////////////////////////////////////////////

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

  __FAILURE_HANDLE(array_clear(allocator->allocated_objects));

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

static LuminaryResult _lum_compiler_stack_allocator_push_member(
  LumCompilerStackAllocator* allocator, uint32_t base_id, size_t offset, LumBuiltinType type, uint32_t* id) {
  __CHECK_NULL_ARGUMENT(allocator);

  const LumMemoryObject base_obj = allocator->allocated_objects[base_id];
  const size_t size              = lum_builtin_types_sizes[type];

  __DEBUG_ASSERT(offset + size <= base_obj.allocation.size);

  LumMemoryObject object;
  object.type              = type;
  object.allocation.offset = base_obj.allocation.offset + offset;
  object.allocation.size   = size;

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
  LumCompilerConstantAllocator* constant_allocator;
  LumCompilerStackAllocator* stack_allocator;
  LumTokenizer* tokenizer;
  ARRAY LumInstruction* instructions_main;
  ARRAY LumInstruction* instructions_cleanup;
} typedef LumCompilerState;

static LuminaryResult _lum_compiler_state_create(LumCompilerState** state) {
  __CHECK_NULL_ARGUMENT(state);

  __FAILURE_HANDLE(host_malloc(state, sizeof(LumCompilerState)));
  memset(*state, 0, sizeof(LumCompilerState));

  (*state)->stack_ptr = LUM_COMPILER_CONTEXT_STACK_EMPTY;

  __FAILURE_HANDLE(array_create(&(*state)->messages, sizeof(LumCompilerMessage), 16));
  __FAILURE_HANDLE(array_create(&(*state)->instructions_main, sizeof(LumInstruction), 16));
  __FAILURE_HANDLE(array_create(&(*state)->instructions_cleanup, sizeof(LumInstruction), 16));

  __FAILURE_HANDLE(_lum_compiler_constant_allocator_create(&(*state)->constant_allocator));
  __FAILURE_HANDLE(_lum_compiler_stack_allocator_create(&(*state)->stack_allocator));
  __FAILURE_HANDLE(lum_tokenizer_create(&(*state)->tokenizer));

  return LUMINARY_SUCCESS;
}

static LumCompilerContextType _lum_compiler_state_get_current_context_type(const LumCompilerState* state) {
  return (state->stack_ptr == LUM_COMPILER_CONTEXT_STACK_EMPTY) ? LUM_COMPILER_CONTEXT_TYPE_NULL
                                                                : state->context_stack[state->stack_ptr].type;
}

static LuminaryResult _lum_compiler_state_add_info_message(LumCompilerState* state, const LumToken* token, const char* format, ...) {
  LumCompilerMessage message;
  __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_INFO, token));

  va_list args;
  va_start(args, format);
  vsnprintf(message.message, LUM_COMPILER_MAX_MESSAGE_LENGTH, format, args);
  va_end(args);

  __FAILURE_HANDLE(array_push(&state->messages, &message));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_state_add_warn_message(LumCompilerState* state, const LumToken* token, const char* format, ...) {
  LumCompilerMessage message;
  __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_WARNING, token));

  va_list args;
  va_start(args, format);
  vsnprintf(message.message, LUM_COMPILER_MAX_MESSAGE_LENGTH, format, args);
  va_end(args);

  __FAILURE_HANDLE(array_push(&state->messages, &message));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_state_add_error_message(LumCompilerState* state, const LumToken* token, const char* format, ...) {
  LumCompilerMessage message;
  __FAILURE_HANDLE(_lum_compiler_message_init(&message, LUM_COMPILER_MESSAGE_TYPE_ERROR, token));

  va_list args;
  va_start(args, format);
  vsnprintf(message.message, LUM_COMPILER_MAX_MESSAGE_LENGTH, format, args);
  va_end(args);

  state->error_occurred = true;

  __FAILURE_HANDLE(array_push(&state->messages, &message));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_state_destroy(LumCompilerState** state) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(*state);

  __FAILURE_HANDLE(lum_tokenizer_destroy(&(*state)->tokenizer));
  __FAILURE_HANDLE(_lum_compiler_stack_allocator_destroy(&(*state)->stack_allocator));
  __FAILURE_HANDLE(_lum_compiler_constant_allocator_destroy(&(*state)->constant_allocator));

  __FAILURE_HANDLE(array_destroy(&(*state)->messages));
  __FAILURE_HANDLE(array_destroy(&(*state)->instructions_main));
  __FAILURE_HANDLE(array_destroy(&(*state)->instructions_cleanup));

  __FAILURE_HANDLE(host_free(state));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Instructions
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_emit_ldg(LumCompilerState* state, uint32_t dst_stack_id, uint32_t src_const_id, bool is_cleanup) {
  __CHECK_NULL_ARGUMENT(state);

  LumMemoryObject dst_object = state->stack_allocator->allocated_objects[dst_stack_id];

  const bool dst_is_addressable = lum_builtin_types_addressable[dst_object.type];

  LumMemoryAllocation src_allocation;
  if (dst_is_addressable) {
    LumMemoryObject src_object;
    __FAILURE_HANDLE(_lum_compiler_constant_allocator_get_memory_object(state->constant_allocator, src_const_id, &src_object));

    __DEBUG_ASSERT(src_object.type == LUM_BUILTIN_TYPE_STRING);

    src_allocation = src_object.allocation;
  }
  else {
    src_allocation = (LumMemoryAllocation) {.offset = 0, .size = 0};
  }

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_ldg(&instruction, dst_object.type, dst_object.allocation, src_allocation));

  if (is_cleanup == false) {
    __FAILURE_HANDLE(array_push(&state->instructions_main, &instruction));
  }
  else {
    __FAILURE_HANDLE(array_push(&state->instructions_cleanup, &instruction));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_stg(LumCompilerState* state, uint32_t src_stack_id, bool is_cleanup) {
  __CHECK_NULL_ARGUMENT(state);

  LumMemoryObject src_object = state->stack_allocator->allocated_objects[src_stack_id];

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_stg(&instruction, src_object.type, src_object.allocation));

  if (is_cleanup == false) {
    __FAILURE_HANDLE(array_push(&state->instructions_main, &instruction));
  }
  else {
    __FAILURE_HANDLE(array_push(&state->instructions_cleanup, &instruction));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_emit_mov(LumCompilerState* state, uint32_t dst_stack_id, uint32_t src_id, bool is_cleanup) {
  __CHECK_NULL_ARGUMENT(state);

  LumMemoryObject dst_object = state->stack_allocator->allocated_objects[dst_stack_id];
  LumMemoryObject src_object = (src_id & ALLOCATOR_OBJECT_ID_CONSTANT)
                                 ? state->constant_allocator->allocated_objects[src_id & ALLOCATOR_OBJECT_ID_MASK].obj
                                 : state->stack_allocator->allocated_objects[src_id & ALLOCATOR_OBJECT_ID_MASK];

  LumInstruction instruction;
  __FAILURE_HANDLE(lum_instruction_encode_mov(&instruction, dst_object.type, dst_object.allocation, src_object.allocation));

  if (is_cleanup == false) {
    __FAILURE_HANDLE(array_push(&state->instructions_main, &instruction));
  }
  else {
    __FAILURE_HANDLE(array_push(&state->instructions_cleanup, &instruction));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_null_binary(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  __FAILURE_HANDLE(array_clear(binary->instructions));

  binary->stack_size           = 0;
  binary->constant_memory_size = 0;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Context resolution
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_context_resolve(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  if (state->stack_ptr == LUM_COMPILER_CONTEXT_STACK_EMPTY) {
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "internal compiler error, resolved empty stack context"));
    return LUMINARY_SUCCESS;
  }

  LumCompilerContext* context = state->context_stack + state->stack_ptr;

  __DEBUG_ASSERT(context->type != LUM_COMPILER_CONTEXT_TYPE_NULL);

  switch (context->type) {
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS: {
      if (context->access.type == LUM_BUILTIN_TYPE_VOID) {
        __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "expected type identifier before ']'"));
        return LUMINARY_SUCCESS;
      }

      const bool is_addressable = lum_builtin_types_addressable[context->access.type];

      if (is_addressable && context->access.string_constant_object_id == ALLOCATOR_OBJECT_ID_INVALID) {
        __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "expected string literal before ']'"));
        return LUMINARY_SUCCESS;
      }

      if (is_addressable == false && context->access.string_constant_object_id != ALLOCATOR_OBJECT_ID_INVALID) {
        __FAILURE_HANDLE(_lum_compiler_state_add_error_message(
          state, token, "type '%s' is not addressable", lum_builtin_types_strings[context->access.type]));
        return LUMINARY_SUCCESS;
      }

      uint32_t dst_stack_id;
      __FAILURE_HANDLE(_lum_compiler_stack_allocator_push(state->stack_allocator, context->access.type, &dst_stack_id));

      __FAILURE_HANDLE(_lum_compiler_emit_ldg(state, dst_stack_id, context->access.string_constant_object_id, false));
      __FAILURE_HANDLE(_lum_compiler_emit_stg(state, dst_stack_id, true));

      state->returned_stack_object_id = dst_stack_id;
    } break;
    case LUM_COMPILER_CONTEXT_TYPE_MEMBER_ACCESS: {
      state->returned_stack_object_id = context->member_access.member_stack_object_id;
    } break;
    case LUM_COMPILER_CONTEXT_TYPE_OPERATOR: {
      __FAILURE_HANDLE(_lum_compiler_emit_mov(state, context->operator.dst_stack_object_id, context->operator.src_object_id, false));
    } break;
    default:
      break;
  }

  state->stack_ptr--;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_context_finalize_statement(LumCompilerState* state) {
  __CHECK_NULL_ARGUMENT(state);

  __FAILURE_HANDLE(array_append(&state->binary->instructions, state->instructions_main));
  __FAILURE_HANDLE(array_append(&state->binary->instructions, state->instructions_cleanup));

  __FAILURE_HANDLE(array_clear(state->instructions_main));
  __FAILURE_HANDLE(array_clear(state->instructions_cleanup));

  if (state->stack_allocator->allocated_stack_size > state->binary->stack_size)
    state->binary->stack_size = state->stack_allocator->allocated_stack_size;

  __FAILURE_HANDLE(_lum_compiler_stack_allocator_reset(state->stack_allocator));

  state->returned_stack_object_id = ALLOCATOR_OBJECT_ID_INVALID;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Identifier
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_compiler_handle_identifier_access_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  if (token->identifier.is_builtin_type == false) {
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "'%s' does not name a type", token->identifier.name));
    return LUMINARY_SUCCESS;
  }

  if (token->identifier.builtin_type == LUM_BUILTIN_TYPE_VOID) {
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "incomplete type 'void' is not allowed"));
    return LUMINARY_SUCCESS;
  }

  state->context_stack[state->stack_ptr].access.type = token->identifier.builtin_type;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_identifier_member_access_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  const uint32_t base_stack_object_id = state->context_stack[state->stack_ptr].member_access.base_stack_object_id;

  __DEBUG_ASSERT(base_stack_object_id != ALLOCATOR_OBJECT_ID_INVALID);

  const LumBuiltinType base_type = state->stack_allocator->allocated_objects[base_stack_object_id].type;

  bool valid_member_name                      = false;
  const LumBuiltinTypeMember* selected_member = (const LumBuiltinTypeMember*) 0;

  if (token->identifier.is_builtin_type == false) {
    const uint32_t num_members          = lum_builtin_types_member_counts[base_type];
    const LumBuiltinTypeMember* members = lum_builtin_types_member[base_type];

    for (uint32_t member_id = 0; member_id < num_members; member_id++) {
      if (strcmp(token->identifier.name, members[member_id].name) == 0) {
        selected_member   = members + member_id;
        valid_member_name = true;
        break;
      }
    }
  }

  if (valid_member_name == false) {
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(
      state, token, "'%s' is not a member of '%s'", token->identifier.name, lum_builtin_types_strings[base_type]));
    return LUMINARY_SUCCESS;
  }

  uint32_t member_stack_id;
  __FAILURE_HANDLE(_lum_compiler_stack_allocator_push_member(
    state->stack_allocator, base_stack_object_id, selected_member->offset, selected_member->type, &member_stack_id));

  state->context_stack[state->stack_ptr].member_access.member_stack_object_id = member_stack_id;

  __FAILURE_HANDLE(_lum_compiler_context_resolve(state, token));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_identifier(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL:
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected identifier"));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_handle_identifier_access_context(state, token));
      break;

    case LUM_COMPILER_CONTEXT_TYPE_MEMBER_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_handle_identifier_member_access_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_OPERATOR:
    case LUM_COMPILER_CONTEXT_TYPE_INITIALIZER:
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected identifier"));
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
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "expected string literal"));
    return LUMINARY_SUCCESS;
  }

  uint32_t constant_id;
  __FAILURE_HANDLE(_lum_compiler_constant_allocator_push_literal(state->constant_allocator, &token->literal, &constant_id));

  state->context_stack[state->stack_ptr].access.string_constant_object_id = constant_id;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_literal_operator_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  const uint32_t lhs_stack_id = state->context_stack[state->stack_ptr].operator.dst_stack_object_id;

  __DEBUG_ASSERT(lhs_stack_id != ALLOCATOR_OBJECT_ID_INVALID);

  const LumBuiltinType lhs_type = state->stack_allocator->allocated_objects[lhs_stack_id].type;
  const LumBuiltinType rhs_type = lum_tokenizer_literal_type_to_builtin[token->literal.type];

  if (lhs_type != rhs_type) {
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(
      state, token, "a value of type '%s' cannot be assigned to an entity of type '%s'", lum_builtin_types_strings[rhs_type],
      lum_builtin_types_strings[lhs_type]));
    return LUMINARY_SUCCESS;
  }

  uint32_t constant_id;
  __FAILURE_HANDLE(_lum_compiler_constant_allocator_push_literal(state->constant_allocator, &token->literal, &constant_id));

  state->context_stack[state->stack_ptr].operator.src_object_id = constant_id;

  __FAILURE_HANDLE(_lum_compiler_context_resolve(state, token));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_literal(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  LumCompilerContextType type = _lum_compiler_state_get_current_context_type(state);

  switch (type) {
    case LUM_COMPILER_CONTEXT_TYPE_NULL:
      __FAILURE_HANDLE(_lum_compiler_state_add_warn_message(state, token, "unexpected literal"));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_handle_literal_access_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_MEMBER_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected literal"));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_OPERATOR:
      __FAILURE_HANDLE(_lum_compiler_handle_literal_operator_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_INITIALIZER:
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected literal"));
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

  if (type != LUM_COMPILER_CONTEXT_TYPE_NULL && type != LUM_COMPILER_CONTEXT_TYPE_INITIALIZER) {
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected operator"));
    return LUMINARY_SUCCESS;
  }

  LumCompilerContext context;
  context.type = LUM_COMPILER_CONTEXT_TYPE_OPERATOR;

  if (state->returned_stack_object_id == ALLOCATOR_OBJECT_ID_INVALID) {
    __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected operator"));
    return LUMINARY_SUCCESS;
  }

  context.operator.dst_stack_object_id = state->returned_stack_object_id;
  context.operator.src_object_id       = ALLOCATOR_OBJECT_ID_INVALID;

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
    case LUM_SEPARATOR_TYPE_STATEMENT_END:
      __FAILURE_HANDLE(_lum_compiler_context_finalize_statement(state));
      break;
    case LUM_SEPARATOR_TYPE_ACCESS_BEGIN: {
      LumCompilerContext context;
      context.type                             = LUM_COMPILER_CONTEXT_TYPE_ACCESS;
      context.access.type                      = LUM_BUILTIN_TYPE_VOID;
      context.access.string_constant_object_id = ALLOCATOR_OBJECT_ID_INVALID;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of accessor"));
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected member separator"));
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected list separator"));
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN: {
      if (state->returned_stack_object_id == ALLOCATOR_OBJECT_ID_INVALID) {
        __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "cannot initialize 'null' object"));
        break;
      }

      LumCompilerContext context;
      context.type                        = LUM_COMPILER_CONTEXT_TYPE_INITIALIZER;
      context.initializer.stack_object_id = state->returned_stack_object_id;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of initializer"));
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator_access_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_STATEMENT_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of statement"));
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_BEGIN: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected begin of accessor"));
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_END: {
      __FAILURE_HANDLE(_lum_compiler_context_resolve(state, token));
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected member separator"));
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected list separator"));
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected begin of initializer"));
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of initializer"));
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
    case LUM_SEPARATOR_TYPE_STATEMENT_END:
      __FAILURE_HANDLE(_lum_compiler_context_resolve(state, token));
      break;
    case LUM_SEPARATOR_TYPE_ACCESS_BEGIN: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected begin of accessor"));
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of accessor"));
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected member_separator"));
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected list separato"));
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected begin of initializer"));
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of initializer"));
    } break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_compiler_handle_separator_initializer_context(LumCompilerState* state, const LumToken* token) {
  __CHECK_NULL_ARGUMENT(state);
  __CHECK_NULL_ARGUMENT(token);

  switch (token->separator.type) {
    case LUM_SEPARATOR_TYPE_STATEMENT_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of statement"));
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_BEGIN: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected begin of accessor"));
    } break;
    case LUM_SEPARATOR_TYPE_ACCESS_END: {
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected end of accessor"));
    } break;
    case LUM_SEPARATOR_TYPE_MEMBER: {
      uint32_t base_stack_id = state->context_stack[state->stack_ptr].initializer.stack_object_id;

      __DEBUG_ASSERT(base_stack_id != ALLOCATOR_OBJECT_ID_INVALID);

      LumCompilerContext context;
      context.type                               = LUM_COMPILER_CONTEXT_TYPE_MEMBER_ACCESS;
      context.member_access.base_stack_object_id = base_stack_id;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_LIST: {
      state->returned_stack_object_id = ALLOCATOR_OBJECT_ID_INVALID;
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_BEGIN: {
      if (state->returned_stack_object_id == ALLOCATOR_OBJECT_ID_INVALID) {
        __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected begin of initializer"));
        break;
      }

      LumCompilerContext context;
      context.type                        = LUM_COMPILER_CONTEXT_TYPE_INITIALIZER;
      context.initializer.stack_object_id = state->returned_stack_object_id;

      state->context_stack[++state->stack_ptr] = context;
    } break;
    case LUM_SEPARATOR_TYPE_INITIALIZER_END: {
      __FAILURE_HANDLE(_lum_compiler_context_resolve(state, token));
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
    case LUM_COMPILER_CONTEXT_TYPE_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_access_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_MEMBER_ACCESS:
      __FAILURE_HANDLE(_lum_compiler_state_add_error_message(state, token, "unexpected separator"));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_OPERATOR:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_operator_context(state, token));
      break;
    case LUM_COMPILER_CONTEXT_TYPE_INITIALIZER:
      __FAILURE_HANDLE(_lum_compiler_handle_separator_initializer_context(state, token));
      break;
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

  state->returned_stack_object_id = ALLOCATOR_OBJECT_ID_INVALID;
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

    if (state->error_occurred)
      break;
  } while (token.type != LUM_TOKEN_TYPE_EOF);

  uint32_t message_count;
  __FAILURE_HANDLE(array_get_num_elements(state->messages, &message_count));

  uint32_t num_errors = 0;

  for (uint32_t message_id = 0; message_id < message_count; message_id++) {
    const LumCompilerMessage* message = state->messages + message_id;

    switch (message->type) {
      case LUM_COMPILER_MESSAGE_TYPE_INFO:
        if (compiler->log_level >= 2) {
          info_message("<source>:%u:%u: info: %s", message->line, message->col, message->message);
        }
        break;
      case LUM_COMPILER_MESSAGE_TYPE_WARNING:
        if (compiler->log_level >= 1) {
          warn_message("<source>:%u:%u: warn: %s", message->line, message->col, message->message);
        }
        break;
      case LUM_COMPILER_MESSAGE_TYPE_ERROR:
        error_message("<source>:%u:%u: error: %s", message->line, message->col, message->message);
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

  __FAILURE_HANDLE(_lum_compiler_constant_allocator_create_data_section(state->constant_allocator, state->binary));
  __FAILURE_HANDLE(_lum_compiler_constant_allocator_reset(state->constant_allocator));

  if (state->error_occurred) {
    __FAILURE_HANDLE(_lum_compiler_null_binary(state->binary));
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Failed to compile.");
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compiler_destroy(LumCompiler** compiler) {
  __CHECK_NULL_ARGUMENT(compiler);
  __CHECK_NULL_ARGUMENT(*compiler);

  __FAILURE_HANDLE(_lum_compiler_state_destroy((LumCompilerState**) &(*compiler)->data));

  __FAILURE_HANDLE(host_free(compiler));

  return LUMINARY_SUCCESS;
}
