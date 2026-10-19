// Link the unchanged production TUs: no copied reduction kernel or model mocks.
#include "strata/core/remote_expert_opt.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void checked(cudaError_t e, const char* expr, int line) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(expr) + " line=" + std::to_string(line) + ": " + cudaGetErrorString(e));
}
#define ck(expr) checked((expr), #expr, __LINE__)
static void require(bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); }

namespace strata::core {
struct RemoteReduceTestAccess {
    static int run(int dev, bool enabled) {
        constexpr int H = kernels::cpu::H, K = 10, T = 8, REPS = 5;
        ck(cudaSetDevice(dev));
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, dev));
        require(prop.canMapHostMemory, "device cannot map host memory");
        size_t free_before, total;
        ck(cudaMemGetInfo(&free_before, &total));
        RemoteExperts remote;
        // Destruction order matters: the opt must detach before remote closes.
        RemoteExpertOpt opt;
        remote.device_ = dev;
        ck(cudaStreamCreateWithFlags(&remote.stream_, cudaStreamNonBlocking));
        ck(cudaMalloc((void**) &remote.d_out_, T * K * H * sizeof(float)));
        ck(cudaHostAlloc((void**) &remote.h_out_, T * H * sizeof(float),
                         cudaHostAllocMapped | cudaHostAllocPortable));
        ck(cudaHostGetDevicePointer((void**) &remote.z_out_, remote.h_out_, 0));
        const float* alias = remote.z_out_;
        ck(cudaMalloc(&remote.d_meta_, RemoteExpertOpt::metadata_bytes()));
        ck(cudaHostAlloc(&remote.h_meta_, RemoteExpertOpt::metadata_bytes(), cudaHostAllocPortable));
        remote.owned_.resize(T * K);
        // Exercise real finish() synchronization/accumulate, even for empty metadata.
        remote.group_id_.push_back(0);
        opt.attach(remote);
        std::string err;
        require(opt.init(err), err);
        std::vector<float> parts(T * K * H), weights(T * K), expected(T * H), scratch(T * H);
        int cases = 0, graph_checks = 0, graph_skips = 0;
        uint64_t hash = 1469598103934665603ULL;
        for (int zc : {1, 0}) {
            remote.zero_copy_ = zc;
            // If fallback accidentally uses mapped output, a null alias catches it.
            remote.z_out_ = zc ? const_cast<float*>(alias) : nullptr;
            const bool mapped = enabled && zc;
            for (int tokens = 1; tokens <= T; ++tokens) {
                for (int kind = 0; kind < 3; ++kind) {
                    for (int graph_mode = 0; graph_mode < 2; ++graph_mode) {
                        cudaGraph_t graph = nullptr;
                        cudaGraphExec_t exec = nullptr;
                        for (int rep = 0; rep < REPS; ++rep) {
                            remote.original_row_.clear();
                            std::fill(remote.owned_.begin(), remote.owned_.end(), 0);
                            // Empty, sparse with changing holes, full with a changed compact order.
                            for (int i = 0; i < tokens * K; ++i) {
                                weights[i] = float(((i * 3 + rep * 5) % 17) - 8) / 16.0f;
                                if (kind == 2 || (kind == 1 && (i + rep) % 3 == 0)) {
                                    remote.original_row_.push_back(i);
                                    remote.owned_[i] = 1;
                                }
                            }
                            std::reverse(remote.original_row_.begin(), remote.original_row_.end());
                            for (size_t i = 0; i < parts.size(); ++i)
                                parts[i] = float(int((i * 13 + rep * 29 + tokens * 7 + kind) % 251) - 125) / 32.0f;
                            std::fill(expected.begin(), expected.end(), 0.0f);
                            std::vector<int> row(tokens * K, -1);
                            for (size_t r = 0; r < remote.original_row_.size(); ++r)
                                row[remote.original_row_[r]] = int(r);
                            for (int t = 0; t < tokens; ++t)
                                for (int c = 0; c < H; ++c)
                                    for (int j = 0; j < K; ++j) {
                                        const int i = t * K + j;
                                        if (row[i] >= 0) expected[t * H + c] =
                                            std::fma(parts[row[i] * H + c], weights[i], expected[t * H + c]);
                                    }
                            opt.begin(weights.data(), 0, tokens);
                            opt.prepare(remote, remote.h_meta_);
                            ck(cudaMemcpyAsync(remote.d_out_, parts.data(), parts.size() * sizeof(float),
                                               cudaMemcpyHostToDevice, remote.stream_));
                            ck(cudaMemcpyAsync(remote.d_meta_, remote.h_meta_, RemoteExpertOpt::metadata_bytes(),
                                               cudaMemcpyHostToDevice, remote.stream_));
                            ck(cudaMemsetAsync(opt.peers_[0].sum, 0x5a, T * H * sizeof(float), remote.stream_));
                            ck(cudaStreamSynchronize(remote.stream_));
                            std::memset(remote.h_out_, 0xa5, T * H * sizeof(float));
                            if (graph_mode && rep == 0) {
                                auto capture = cudaStreamBeginCapture(remote.stream_, cudaStreamCaptureModeThreadLocal);
                                if (capture == cudaErrorStreamCaptureUnsupported || capture == cudaErrorNotSupported) {
                                    cudaGetLastError();
                                    ++graph_skips;
                                    break;
                                }
                                ck(capture);
                                require(opt.reduce(remote, remote.d_meta_, err), err);
                                ck(cudaStreamEndCapture(remote.stream_, &graph));
                                size_t n = 0;
                                ck(cudaGraphGetNodes(graph, nullptr, &n));
                                std::vector<cudaGraphNode_t> nodes(n);
                                ck(cudaGraphGetNodes(graph, nodes.data(), &n));
                                int kernels = 0, copies = 0;
                                for (auto node : nodes) {
                                    cudaGraphNodeType type;
                                    ck(cudaGraphNodeGetType(node, &type));
                                    if (type == cudaGraphNodeTypeKernel) {
                                        ++kernels;
                                        cudaKernelNodeParams params{};
                                        ck(cudaGraphKernelNodeGetParams(node, &params));
                                        float* dst = *static_cast<float**>(params.kernelParams[2]);
                                        require(dst == (mapped ? remote.z_out_ : opt.peers_[0].sum),
                                                "PATH FAIL: production reduce kernel has wrong output pointer");
                                    } else if (type == cudaGraphNodeTypeMemcpy) ++copies;
                                }
                                require(kernels == 1 && copies == (mapped ? 0 : 1),
                                        "PATH FAIL: production graph kernel/copy count");
                                ++graph_checks;
                                ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                            }
                            if (graph_mode) ck(cudaGraphLaunch(exec, remote.stream_));
                            else require(opt.reduce(remote, remote.d_meta_, err), err);
                            require(remote.finish(nullptr, err), err);
                            ck(cudaGetLastError());
                            require(std::memcmp(expected.data(), remote.h_out_, tokens * H * sizeof(float)) == 0,
                                    "PARITY FAIL: CPU fma vs production reduce output");
                            require(std::memcmp(expected.data(), opt.h_sum_, tokens * H * sizeof(float)) == 0,
                                    "PARITY FAIL: finish/accumulate output");
                            ck(cudaMemcpy(scratch.data(), opt.peers_[0].sum, T * H * sizeof(float), cudaMemcpyDeviceToHost));
                            if (mapped) {
                                const auto* bytes = reinterpret_cast<const unsigned char*>(scratch.data());
                                require(std::all_of(bytes, bytes + T * H * sizeof(float),
                                                    [](unsigned char c) { return c == 0x5a; }),
                                        "PATH FAIL: mapped enabled but production reduce still writes device sum (missing new path)");
                            } else require(std::memcmp(expected.data(), scratch.data(), tokens * H * sizeof(float)) == 0,
                                           "PATH FAIL: fallback did not use device sum");
                            const auto* bytes = reinterpret_cast<const unsigned char*>(remote.h_out_);
                            for (size_t i = 0; i < tokens * H * sizeof(float); ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
                            // Guard against overrunning the requested token width.
                            require(std::all_of(bytes + tokens * H * sizeof(float), bytes + T * H * sizeof(float),
                                                [](unsigned char c) { return c == 0xa5; }), "output tail overwritten");
                            ++cases;
                            opt.end();
                        }
                        if (exec) ck(cudaGraphExecDestroy(exec));
                        if (graph) ck(cudaGraphDestroy(graph));
                    }
                }
            }
        }
        size_t free_after;
        ck(cudaMemGetInfo(&free_after, &total));
        const size_t delta = free_before > free_after ? free_before - free_after : 0;
        std::printf("PASS device=%d name=%s enabled=%d cases=%d graph_checks=%d graph_skips=%d hash=%016llx device_delta_bytes=%zu\n",
                    dev, prop.name, enabled, cases, graph_checks, graph_skips, (unsigned long long) hash, delta);
        require(delta < 100ull * 1024 * 1024, "test device memory exceeds 100 MiB target");
        require(cases == 480 - graph_skips * REPS, "unexpected completed test count");
        return cases;
    }
};
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: remote_reduce_parity DEVICE EXPECT_ENABLED(0|1); set env externally");
        const int dev = std::stoi(argv[1]);
        const bool enabled = std::stoi(argv[2]) != 0;
        strata::core::RemoteReduceTestAccess::run(dev, enabled);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
