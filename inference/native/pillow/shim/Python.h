/* Build-only adapter for libImaging's ownership hooks. No interpreter ABI. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#define HAVE_PROTOTYPES 1
#define STDC_HEADERS 1
typedef ptrdiff_t Py_ssize_t;
typedef struct ImagineOpaquePythonObject PyObject;
void imagine_pillow_error_clear(void);
#define PyErr_Clear imagine_pillow_error_clear
/* Arrow objects cannot enter the buffer-only exported ABI. Their original
   implementations remain unmodified and are removed as unreachable sections. */
void imagine_pillow_arrow_unavailable(void);
#define Py_INCREF(x) imagine_pillow_arrow_unavailable()
#define Py_DECREF(x) imagine_pillow_arrow_unavailable()
void *PyCapsule_GetPointer(PyObject *, const char *);
