// Diagnostic LD_PRELOAD library. No synchronization or inference changes.
// Built against the installed CUDA 13.4 CUPTI record layouts.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <set>
#include <execinfo.h>
#include <dlfcn.h>

struct Trace { FILE* file; std::mutex mutex; size_t dropped = 0; };
static Trace* trace;
static FILE* stacks;
static CUpti_SubscriberHandle subscriber;
static std::set<size_t>* sampled_sizes;
static void check(CUptiResult result) {
    if (result == CUPTI_SUCCESS) return;
    const char* message = nullptr;
    cuptiGetResultString(result, &message);
    std::fprintf(stderr, "CUPTI activity error: %s\n", message ? message : "unknown");
    std::_Exit(2);
}
static uint64_t CUPTIAPI timestamp() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return uint64_t(now.tv_sec) * 1000000000ULL + now.tv_nsec;
}
static void CUPTIAPI copy_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* raw) {
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
    const auto* data = static_cast<const CUpti_CallbackData*>(raw);
    if (data->callbackSite != CUPTI_API_ENTER) return;
    size_t bytes = 0, width = 0, height = 0, spitch = 0, dpitch = 0;
    if (id == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020) {
        auto* p = static_cast<const cudaMemcpyAsync_v3020_params*>(data->functionParams);
        if (p->kind != cudaMemcpyDeviceToDevice) return;
        bytes = p->count;
    } else if (id == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020) {
        auto* p = static_cast<const cudaMemcpy2DAsync_v3020_params*>(data->functionParams);
        if (p->kind != cudaMemcpyDeviceToDevice) return;
        width = p->width; height = p->height; spitch = p->spitch; dpitch = p->dpitch;
        bytes = width * height;
    } else return;
    if (bytes < 80 * 1024 * 1024) return;
    std::lock_guard<std::mutex> lock(trace->mutex);
    if (!sampled_sizes->insert(bytes).second) return;
    std::fprintf(stacks, "COPY %s bytes=%zu width=%zu height=%zu spitch=%zu dpitch=%zu correlation=%u\n",
                 data->functionName, bytes, width, height, spitch, dpitch, data->correlationId);
    void* frames[40];
    int count = backtrace(frames, 40);
    for (int i = 0; i < count; ++i) {
        Dl_info info{};
        if (dladdr(frames[i], &info))
            std::fprintf(stacks, "FRAME %s 0x%zx\n", info.dli_fname,
                         size_t(static_cast<char*>(frames[i]) - static_cast<char*>(info.dli_fbase)));
    }
    std::fflush(stacks);
}
static void CUPTIAPI request(uint8_t** buffer, size_t* size, size_t* maximum) {
    *size = 8 * 1024 * 1024;
    *maximum = 0;
    *buffer = static_cast<uint8_t*>(std::aligned_alloc(8, *size));
    if (!*buffer) std::_Exit(2);
}
static void CUPTIAPI complete(CUcontext context, uint32_t stream, uint8_t* buffer,
                              size_t, size_t valid) {
    std::lock_guard<std::mutex> lock(trace->mutex);
    CUpti_Activity* record = nullptr;
    while (true) {
        auto result = cuptiActivityGetNextRecord(buffer, valid, &record);
        if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        check(result);
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* k = reinterpret_cast<CUpti_ActivityKernel13*>(record);
            std::fprintf(trace->file, "kernel\t%llu\t%llu\t%u\t%u\t0\t%d,%d,%d\t%d,%d,%d\t%s\n",
                (unsigned long long)k->start, (unsigned long long)k->end,
                k->streamId, k->correlationId, k->gridX, k->gridY, k->gridZ,
                k->blockX, k->blockY, k->blockZ, k->name);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto* m = reinterpret_cast<CUpti_ActivityMemcpy7*>(record);
            std::fprintf(trace->file, "memcpy\t%llu\t%llu\t%u\t%u\t%llu\t-\t-\tcopy_kind_%u\n",
                (unsigned long long)m->start, (unsigned long long)m->end,
                m->streamId, m->correlationId, (unsigned long long)m->bytes, m->copyKind);
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            auto* m = reinterpret_cast<CUpti_ActivityMemset5*>(record);
            std::fprintf(trace->file, "memset\t%llu\t%llu\t%u\t%u\t%llu\t-\t-\tmemset\n",
                (unsigned long long)m->start, (unsigned long long)m->end,
                m->streamId, m->correlationId, (unsigned long long)m->bytes);
        }
    }
    size_t dropped = 0;
    check(cuptiActivityGetNumDroppedRecords(context, stream, &dropped));
    trace->dropped += dropped;
    std::free(buffer);
}
__attribute__((constructor)) static void start() {
    const char* path = std::getenv("QWEN_ACTIVITY_TRACE");
    if (!path) return;
    trace = new Trace;
    trace->file = std::fopen(path, "w");
    if (!trace->file) { std::perror("activity trace"); std::_Exit(2); }
    std::fprintf(trace->file, "kind\tstart_ns\tend_ns\tstream\tcorrelation\tbytes\tgrid\tblock\tname\n");
    check(cuptiActivityRegisterTimestampCallback(timestamp));
    check(cuptiActivityRegisterCallbacks(request, complete));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    if (const char* stack_path = std::getenv("QWEN_COPY_STACKS")) {
        stacks = std::fopen(stack_path, "w");
        if (!stacks) std::_Exit(2);
        sampled_sizes = new std::set<size_t>;
        check(cuptiSubscribe(&subscriber, copy_callback, nullptr));
        check(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020));
        check(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020));
    }
}
__attribute__((destructor)) static void finish() {
    if (!trace) return;
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    std::fprintf(trace->file, "# dropped_records=%zu\n", trace->dropped);
    std::fclose(trace->file);
    if (stacks) {
        check(cuptiUnsubscribe(subscriber));
        std::fclose(stacks);
    }
}
