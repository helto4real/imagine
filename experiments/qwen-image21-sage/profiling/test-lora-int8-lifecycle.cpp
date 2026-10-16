// Synthetic ModelManager integration: original restore, reuse and LoRA epochs.
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "model_manager.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_lifecycle(const char *model, const char *lora,
                            ggml_backend_t cuda, ggml_backend_t cpu, int mode,
                            bool fused = false) {
  auto manager = std::make_shared<ModelManager>();
  ModelLoader loader;
  assert(loader.init_from_file_and_convert_name(model, "",
                                                VERSION_QWEN_IMAGE_2_1));
  assert(manager->set_loader(std::move(loader)));
  manager->set_n_threads(2);
  auto ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
  auto q = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, 512, fused ? 256 : 128);
  auto scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, fused ? 256 : 128);
  const std::string key = "model.diffusion_model.transformer_blocks.0." +
      std::string(fused ? "img_mlp.gate_up.weight" : "attn.to_q.weight");
  using Residency = ModelManager::ResidencyMode;
  const auto params = mode == 0 ? cpu : cuda;
  assert(manager->register_param_tensors(
      ModelComponent::Diffusion, {{key, q}},
      mode == 1 || mode == 3 ? Residency::Disk : Residency::ParamBackend,
      cuda, params));
  assert(manager->register_param_tensors(
      ModelComponent::Diffusion, {{key + "_scale", scale}},
      mode == 1 || mode == 4 ? Residency::Disk : Residency::ParamBackend,
      cuda, params));
  assert(manager->validate_registered_tensors());
  std::vector<ggml_tensor *> tensors{q, scale};
  auto sample = [&]() {
    assert(manager->prepare_params(tensors));
    std::vector<uint8_t> result(ggml_nbytes(q) + ggml_nbytes(scale));
    ggml_backend_tensor_get(q, result.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_get(scale, result.data() + ggml_nbytes(q), 0,
                            ggml_nbytes(scale));
    manager->release_compute_backend_params(tensors);
    return result;
  };
  const auto original = sample();
  ModelManager::LoraSpec spec;
  spec.path = lora;
  spec.multiplier = 1;
  spec.required = true;
  assert(manager->set_loras({spec}, VERSION_QWEN_IMAGE_2_1));
  auto merged = sample();
  assert(merged != original);
  assert(sample() == merged);
  manager->evict_compute_backend_params({q});
  assert(sample() == merged);
  manager->evict_compute_backend_params({scale});
  assert(sample() == merged);
  manager->evict_compute_backend_params(tensors);
  assert(sample() == merged);
  spec.multiplier = -.5f;
  assert(manager->set_loras({spec}, VERSION_QWEN_IMAGE_2_1));
  auto negative = sample();
  assert(negative != merged && negative != original);
  spec.multiplier = 1;
  assert(manager->set_loras({spec}, VERSION_QWEN_IMAGE_2_1));
  assert(sample() == merged);
  assert(manager->set_loras({}, VERSION_QWEN_IMAGE_2_1));
  assert(sample() == original);
  manager.reset();
  ggml_free(ctx);
  std::printf("PASS: synthetic lifecycle residency mode %d fused=%d\n", mode, fused);
}

int main(int argc, char **argv) {
  assert(argc == 6);
  setenv("SD_QWEN21_MERGE_LORA_INT8", "1", 1);
  auto cuda = ggml_backend_cuda_init(0);
  auto cpu = ggml_backend_cpu_init();
  assert(cuda && cpu);
  for (int mode = 0; mode < 5; ++mode) {
    check_lifecycle(argv[1], argv[2], cuda, cpu, mode);
    check_lifecycle(argv[3], argv[4], cuda, cpu, mode, true);
    check_lifecycle(argv[3], argv[5], cuda, cpu, mode, true);
  }
  ggml_backend_free(cuda);
  ggml_backend_free(cpu);
  std::puts("PASS: synthetic merge, reuse, eviction, strength change, restore "
            "and removal");
}
