// Development compiler boundary. Original helper algorithms are generated from
// the pinned source, not replaced by opt/llc defaults or a different compiler.
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <regex>
#include <unordered_set>
#include <unistd.h>
#include "original_triton_llvm.inc"

int main(int argc,char** argv) {
    try {
        if(argc<7) throw std::runtime_error("usage: driver optimize|ptx input output sm features ptx-version [external-library ...]");
        const std::string mode=argv[1],proc=argv[4],features=argv[5];
        imagine_init_targets();
        llvm::LLVMContext context;
        llvm::SMDiagnostic diagnostic;
        auto module=llvm::parseIRFile(argv[2],diagnostic,context);
        if(!module) {diagnostic.print(argv[0],llvm::errs());return 2;}
        std::string result;
        if(mode=="optimize") {
            // MLIR's original in-memory translation uses this identifier.
            // LLVM text parsing otherwise replaces it with the input pathname.
            module->setModuleIdentifier("LLVMDialectModule");
            imagine_set_short_ptr();
            imagine_attach_datalayout(module.get(),"nvptx64-nvidia-cuda",proc,features);
            imagine_set_nvvm_reflect_ftz(module.get());
            std::vector<std::string> libraries;
            for(int i=7;i<argc;++i) libraries.emplace_back(argv[i]);
            if(!libraries.empty() && imagine_has_extern_deps(module.get()))
                imagine_link_extern_libs(module.get(),libraries);
            imagine_optimize_module(module.get());
            llvm::raw_string_ostream output(result);output << *module;
        } else if(mode=="ptx") {
            // Original make_llir sets this LLVM process option before
            // make_ptx runs in the same interpreter process. Re-establish
            // that original state in this separate compiler invocation.
            imagine_set_short_ptr();
            // Original translate_to_asm parses an unnamed MemoryBuffer.
            module->setModuleIdentifier("");
            result=translateLLVMIRToASM(*module,"nvptx64-nvidia-cuda",proc,features,
                                       {"nvptx-mad-wide-opt"},true,false);
            // Original make_ptx's three textual substitutions. PTX is ASCII;
            // no host test tool edits the generated assembly afterward.
            const int version=std::stoi(argv[6]);
            result=std::regex_replace(result,std::regex(R"(\.version \d+\.\d+)"),
                                      ".version "+std::to_string(version/10)+"."+std::to_string(version%10));
            result=std::regex_replace(result,std::regex(R"(\.target sm_\d+)"),".target "+proc);
            result=std::regex_replace(result,std::regex(R"(,\s*debug|debug,\s*)"),"");
        } else throw std::runtime_error("unknown compiler mode");
        std::ofstream output(argv[3],std::ios::binary);
        if(!output) throw std::runtime_error("cannot open compiler output");
        output << result;
        if(!output) throw std::runtime_error("cannot write compiler output");
        return 0;
    } catch(const std::invalid_argument& e) {std::cerr << e.what() << '\n';return 3;}
      catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
