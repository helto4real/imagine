// Development-only probe of the locked LibTorch archive reader. No Python ABI.
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <caffe2/serialize/inline_container.h>
#include <iostream>
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  try {
    caffe2::serialize::PyTorchStreamReader reader(argv[1]);
    auto value=torch::jit::readArchiveAndTensors("data","","",std::nullopt,
      std::nullopt,at::Device(at::kCPU),reader);
    std::cout<<value<<"\n";
    return 0;
  } catch(const c10::Error& error) {
    std::cerr<<error.what_without_backtrace()<<"\n";
    return 1;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<"\n";
    return 1;
  }
}
