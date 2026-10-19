// Real-pack, CPU-only scheduler parity. No generated/quantized replacement weights.
// pool_fused_nvfp4_test BLOB... (extract with tools/extract_pool_nvfp4_fixture.py)
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace {
void flag(const char* key, const char* value) {
#ifdef _WIN32
    if (_putenv_s(key, value)) throw std::runtime_error("environment update failed");
#else
    if (setenv(key, value, 1)) throw std::runtime_error("environment update failed");
#endif
}
// Compile on the pre-feature baseline as well: missing instrumentation is an explicit
// runtime RED, never a successful OFF/OFF comparison masquerading as fused parity.
template<class T> auto fused_count(const T& p, int) -> decltype(p.multi_fused) { return p.multi_fused; }
template<class T> int64_t fused_count(const T&, long) { return -1; }
}

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) throw std::runtime_error("usage: pool_fused_nvfp4_test BLOB...");
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(cpu::kNvfp4Type, cpu::kNvfp4Type, cpu::H, cpu::FF, f, err))
        throw std::runtime_error(err);
    if (f.tail_off == 0 || f.bytes != f.tail_off + cpu::kNvfp4Tail)
        throw std::runtime_error("NVFP4 scale-tail geometry missing");
    std::vector<std::vector<uint8_t>> blobs;
    for (int a = 1; a < argc; ++a) {
        std::ifstream in(argv[a], std::ios::binary | std::ios::ate);
        if (!in || in.tellg() != (std::streamoff) f.bytes) throw std::runtime_error("wrong blob length");
        blobs.emplace_back(f.bytes);
        in.seekg(0);
        if (!in.read((char*) blobs.back().data(), (std::streamsize) f.bytes)) throw std::runtime_error("blob read failed");
        float scales[4];
        std::memcpy(scales, blobs.back().data() + f.tail_off, sizeof scales);
        for (int s = 0; s < 3; ++s)
            if (!std::isfinite(scales[s]) || scales[s] <= 0) throw std::runtime_error("invalid real scale tail");
        std::printf("blob=%s bytes=%zu scales=%.9g,%.9g,%.9g\n", argv[a], f.bytes, scales[0], scales[1], scales[2]);
    }
    const int cap = cpu::ExpertPool::kMaxSplitMulti + 1;
    const size_t elems = (size_t) cap * cpu::MAXT * cpu::H;
    std::vector<float> ref(elems), out(elems), x(cpu::H);
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ExpertJobMulti> jobs(cap);
    std::mt19937 rng(2751);
    int windows = 0;
    int64_t fused = 0;
    // Change activations every window, including the windows that follow an empty layer.
    for (bool host : {false, true}) for (int workers : {1, 2, 8, 15}) {
        flag("STRATA_POOL_FUSED", "0");
        auto off = std::make_unique<cpu::ExpertPool>(workers, false, host);
        flag("STRATA_POOL_FUSED", "1");
        auto on = std::make_unique<cpu::ExpertPool>(workers, false, host);
        for (int nt = 1; nt <= cpu::MAXT; ++nt) for (int r = 0; r < 32; ++r) {
            const int small[] = {0, 1, 2, 7, 15};
            const int n = r == 30 ? cap - 1 : r == 31 ? cap : small[r % 5];
            for (int t = 0; t < cpu::MAXT; ++t) {
                for (float& v : x) v = ((int) (rng() % 20001) - 10000) / 4096.0f;
                cpu::native_quant_act(f, x.data(), act[t].data());
            }
            // Inactive slots and n=0 remain guards; active slots MUST be overwritten.
            std::fill(ref.begin(), ref.end(), -12345.0f);
            std::fill(out.begin(), out.end(), -12345.0f);
            for (int e = 0; e < n; ++e) {
                jobs[e] = cpu::ExpertJobMulti{};
                jobs[e].blob = blobs[(size_t) (e + windows) % blobs.size()].data();
                jobs[e].nt = (r & 1) ? 1 + e % nt : nt;
                for (int t = 0; t < jobs[e].nt; ++t) {
                    jobs[e].nact[t] = act[t].data();
                    const size_t offset = ((size_t) e * cpu::MAXT + t) * cpu::H;
                    jobs[e].out[t] = ref.data() + offset;
                    std::fill_n(ref.data() + offset, cpu::H, std::numeric_limits<float>::quiet_NaN());
                    std::fill_n(out.data() + offset, cpu::H, std::numeric_limits<float>::quiet_NaN());
                }
            }
            const auto before_off = fused_count(*off, 0), before_on = fused_count(*on, 0);
            off->run_split_multi_native(f, jobs.data(), n);
            for (int e = 0; e < n; ++e) for (int t = 0; t < jobs[e].nt; ++t)
                jobs[e].out[t] = out.data() + ((size_t) e * cpu::MAXT + t) * cpu::H;
            if ((r % 8) == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            on->run_split_multi_native(f, jobs.data(), n);
            for (float v : ref) if (!std::isfinite(v)) throw std::runtime_error("reference contains NaN/Inf or unwritten output");
            if (std::memcmp(ref.data(), out.data(), elems * sizeof(float)) != 0) {
                std::fprintf(stderr, "MISMATCH workers=%d host=%d nt=%d repeat=%d jobs=%d\n", workers, host, nt, r, n);
                return 1;
            }
            const int want = workers > 1 && n > 0 ? 1 : 0;
            if ((want && (before_off < 0 || before_on < 0)) || fused_count(*off, 0) != before_off ||
                fused_count(*on, 0) - before_on != want) {
                std::fprintf(stderr, "TRACE ASSERTION FAILED: fused path absent/wrong workers=%d jobs=%d off=%lld on=%lld want_delta=%d\n",
                             workers, n, (long long) fused_count(*off, 0), (long long) fused_count(*on, 0), want);
                return 1;
            }
            ++windows;
        }
        fused += fused_count(*on, 0);
        std::printf("PASS workers=%d host=%d windows=256 fused_calls=%lld\n", workers, host, (long long) fused_count(*on, 0));
    }
    std::printf("PASS memcmp_windows=%d layer_calls=%d fused_calls=%lld workers=1,2,8,15 tokens=1..8 jobs=0,1,2,7,15,96,97 host=0,1 fresh_activations=every_window\n",
                windows, 2 * windows, (long long) fused);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
