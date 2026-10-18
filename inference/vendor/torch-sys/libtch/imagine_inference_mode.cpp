// Owned, thread-local LibTorch InferenceMode context; no Python dependency.
#include <torch/torch.h>
#include <c10/core/InferenceMode.h>
#include "torch_api.h"

void* at_imagine_inference_mode_new(int enabled) {
  PROTECT(return new c10::InferenceMode(enabled != 0);)
  return nullptr;
}
void at_imagine_inference_mode_free(void* value) {
  delete static_cast<c10::InferenceMode*>(value);
}
