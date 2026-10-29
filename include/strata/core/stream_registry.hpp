// include/strata/core/stream_registry.hpp - request -1 diagnostics (xrig, output-neutral): the engine's long-lived
// streams by name, so the stall report can say which of them still has work (cudaStreamQuery) when a request stops.
//
// The #267 release queries only the verifier's two streams, so "the GPU finished in 0 ms" says nothing about the
// drafter's, the main or the copy streams; event 4 (25-Oct) hung in the drafter's stream sync with both of
// those idle.  A stream is added where it is created and removed before it is destroyed; nothing here runs on the
// token path (two relaxed stores per create / destroy).  The request sub-step names the silent call a request is in
// between the stages progress_at already sets ("request" covers from the VRAM query to the drafter's K/V restore).
#pragma once

#include <atomic>
#include <cstdint>

#include "strata/core/progress.hpp"

namespace strata::core {

struct StreamSlot {
    std::atomic<void*> stream{nullptr};
    std::atomic<const char*> name{nullptr};
    std::atomic<int> device{-1};
};

constexpr int kStreamSlots = 64;

inline StreamSlot* stream_slots() {
    static StreamSlot s[kStreamSlots];
    return s;
}

/// a stream to report on (null: nothing); a full table drops it (the report says so by its absence, never fails)
inline void stream_register(void* stream, const char* name, int device) {
    if (stream == nullptr) return;
    StreamSlot* s = stream_slots();
    for (int i = 0; i < kStreamSlots; ++i) {
        void* expected = nullptr;
        if (s[i].stream.load(std::memory_order_relaxed) == nullptr &&
            s[i].stream.compare_exchange_strong(expected, stream)) {
            s[i].name.store(name, std::memory_order_relaxed);
            s[i].device.store(device, std::memory_order_release);
            return;
        }
    }
}

/// before cudaStreamDestroy: the stall report never queries a destroyed stream
inline void stream_unregister(void* stream) {
    if (stream == nullptr) return;
    StreamSlot* s = stream_slots();
    for (int i = 0; i < kStreamSlots; ++i)
        if (s[i].stream.load(std::memory_order_acquire) == stream) {
            s[i].device.store(-1, std::memory_order_relaxed);
            s[i].name.store(nullptr, std::memory_order_relaxed);
            s[i].stream.store(nullptr, std::memory_order_release);
        }
}

/// the silent call a request is in (static strings only), and since when (steady ms)
struct RequestStep {
    std::atomic<const char*> what{"none"};
    std::atomic<int64_t> since_ms{0};
};

inline RequestStep& request_step() {
    static RequestStep r;
    return r;
}

inline void request_step_at(const char* what) {
    RequestStep& r = request_step();
    r.what.store(what, std::memory_order_relaxed);
    r.since_ms.store(progress_now_ms(), std::memory_order_relaxed);
}

/// the drafter's last K/V restore (MtpDrafter::kv_restore): the ring blocks it copied and when it began / ended
/// (ended < began: still waiting in its stream sync, the shape event 4 hung in)
struct DrafterRestore {
    std::atomic<int64_t> b0{-1}, b1{-1};
    std::atomic<int64_t> began_ms{0}, ended_ms{0};
    std::atomic<int64_t> count{0};
};

inline DrafterRestore& drafter_restore() {
    static DrafterRestore d;
    return d;
}

}  // namespace strata::core
