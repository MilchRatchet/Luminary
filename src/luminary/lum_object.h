#ifndef LUMINARY_OBJECT_H
#define LUMINARY_OBJECT_H

#define LUMINARY_OBJECT_DECLARATION(struct_name)      \
  struct {                                            \
    uint32_t reference_count;                         \
    LuminaryResult (*destructor)(struct_name * *obj); \
  } _luminary_object_member

#define LUMINARY_RETAIN(obj)                        \
  if (obj != 0) {                                   \
    obj->_luminary_object_member.reference_count++; \
  }

#define LUMINARY_RELEASE(obj)                                          \
  if (obj != 0) {                                                      \
    if ((--obj->_luminary_object_member.reference_count) == 0)         \
      __FAILURE_HANDLE(obj->_luminary_object_member.destructor(&obj)); \
  }

#endif /* LUMINARY_OBJECT_H */
