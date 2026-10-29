#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace strata::core {

// xrig fix (21-Oct): a small pool of full 16 MiB segment mappings that were freed, reused by the next capture.  A
// fresh mapping costs a page fault per 4 KiB when a park first writes it, on the request path (measured ~+0.8 s TTFT
// per turn on DDR4); transparent huge pages avoid the faults but stall in direct compaction on a box whose free RAM
// is fragmented by the pinned expert copy (one test box: +30% session time).  A pooled mapping is already faulted in.  Up
// to STRATA_CONV_SEGMENT_POOL_MIB (default 2048; 0 turns the pool off) stays mapped; beyond it, and whenever the
// caller needs RAM back (release_segment_pool, from the same place that trims malloc's heap), mappings are unmapped
// as before, so the freed memory still goes back to the kernel.
namespace conversation_detail {
inline constexpr size_t pooled_segment_bytes = 16 * 1024 * 1024;
struct SegmentPool {
    std::mutex m;
    std::vector<void*> free;
    size_t hits = 0, misses = 0, unmapped_over_cap = 0, released = 0;   // counts for the park log line
    std::atomic<size_t> direct_unmapped{0};   // capfix: munmaps of segments the pool does not take (other sizes)
    static SegmentPool& get() { static SegmentPool* pool = new SegmentPool; return *pool; }   // leaked: usable at exit
    static size_t cap() {
        static const size_t c = [] {
            const char* v = std::getenv("STRATA_CONV_SEGMENT_POOL_MIB");
            const long long mib = (v && *v) ? std::atoll(v) : 2048;
            return mib > 0 ? (size_t) mib / 16 : size_t(0);   // in segments
        }();
        return c;
    }
};
}  // namespace conversation_detail

// Unmaps pooled segment mappings until at least `bytes` went back to the kernel (all of them by default), so a
// caller short of RAM gives back only its shortfall and the rest of the pool keeps serving captures.  Returns the
// bytes released.
inline size_t release_segment_pool(size_t bytes = SIZE_MAX) {
#if defined(__linux__)
    auto& pool = conversation_detail::SegmentPool::get();
    std::vector<void*> take;
    {
        std::lock_guard<std::mutex> lock(pool.m);
        const size_t seg = conversation_detail::pooled_segment_bytes;
        const size_t n = bytes == SIZE_MAX ? pool.free.size() : std::min(pool.free.size(), (bytes + seg - 1) / seg);
        take.assign(pool.free.end() - (ptrdiff_t) n, pool.free.end());
        pool.free.resize(pool.free.size() - n);
        pool.released += n;
    }
    for (void* p : take) ::munmap(p, conversation_detail::pooled_segment_bytes);
    return take.size() * conversation_detail::pooled_segment_bytes;
#else
    return 0;
#endif
}

// capfix (28-Oct): the bytes the pool holds mapped and faulted in, which the next capture takes before it maps
// anything new: the park pre-check credits them.
inline size_t segment_pool_held_bytes() {
    auto& pool = conversation_detail::SegmentPool::get();
    std::lock_guard<std::mutex> lock(pool.m);
    return pool.free.size() * conversation_detail::pooled_segment_bytes;
}

// capfix (28-Oct): every munmap of a segment mapping since start - the pool's releases, the 16 MiB segments freed past
// its cap, and segments of other sizes freed directly.  ROCr pins pageable memory through KFD userptr ranges for
// large copies, so an munmap can still start an invalidate/restore: the memfix line prints these after every park
// and restore, to pair with qcount's userptr restores.
struct SegmentMunmaps {
    size_t pool_released = 0, over_cap = 0, direct = 0;
    size_t total() const { return pool_released + over_cap + direct; }
};
inline SegmentMunmaps segment_munmaps() {
    auto& pool = conversation_detail::SegmentPool::get();
    std::lock_guard<std::mutex> lock(pool.m);
    return {pool.released, pool.unmapped_over_cap, pool.direct_unmapped.load(std::memory_order_relaxed)};
}

// "pool held/hits/misses/over_cap/released" for a log line (segments, since start).
inline std::string segment_pool_note() {
    auto& pool = conversation_detail::SegmentPool::get();
    std::lock_guard<std::mutex> lock(pool.m);
    return "segment_pool held=" + std::to_string(pool.free.size()) + " hits=" + std::to_string(pool.hits) +
           " misses=" + std::to_string(pool.misses) + " over_cap=" + std::to_string(pool.unmapped_over_cap) +
           " released=" + std::to_string(pool.released);
}

// xrig fix (20-Oct): on Linux a segment of 1 MiB or more is its own anonymous mapping, unmapped when freed.
// From malloc's heap a freed segment never went back to the kernel: HIP locks pageable copies through KFD SVM
// ranges, and the SVM restore worker faults trimmed (MADV_DONTNEED) heap pages straight back in, so evicting a
// parked conversation left MemAvailable where it was.  munmap ends the range instead.  Other platforms and small
// segments keep the ordinary allocator.
template<class T> struct SegmentAllocator {
    using value_type = T;
    static constexpr size_t map_min = 1024 * 1024;
    SegmentAllocator() = default;
    template<class U> SegmentAllocator(const SegmentAllocator<U>&) noexcept {}
    T* allocate(size_t n) {
#if defined(__linux__)
        if (n * sizeof(T) >= map_min) {
            if (n * sizeof(T) == conversation_detail::pooled_segment_bytes) {   // xrig fix (21-Oct): reuse a pooled one
                auto& pool = conversation_detail::SegmentPool::get();
                std::lock_guard<std::mutex> lock(pool.m);
                if (!pool.free.empty()) { void* q = pool.free.back(); pool.free.pop_back(); ++pool.hits; return static_cast<T*>(q); }
                ++pool.misses;
            }
            void* p = ::mmap(nullptr, n * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) throw std::bad_alloc();
            return static_cast<T*>(p);
        }
#endif
        return std::allocator<T>().allocate(n);
    }
    void deallocate(T* p, size_t n) noexcept {
#if defined(__linux__)
        if (n * sizeof(T) == conversation_detail::pooled_segment_bytes) {   // xrig fix (21-Oct): keep it for reuse
            auto& pool = conversation_detail::SegmentPool::get();
            std::lock_guard<std::mutex> lock(pool.m);
            if (pool.free.size() < conversation_detail::SegmentPool::cap()) { pool.free.push_back(p); return; }
            ++pool.unmapped_over_cap;
        }
        if (n * sizeof(T) >= map_min) {
            if (n * sizeof(T) != conversation_detail::pooled_segment_bytes)   // capfix: past the cap is over_cap
                conversation_detail::SegmentPool::get().direct_unmapped.fetch_add(1, std::memory_order_relaxed);
            ::munmap(p, n * sizeof(T));
            return;
        }
#endif
        std::allocator<T>().deallocate(p, n);
    }
    template<class U> bool operator==(const SegmentAllocator<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const SegmentAllocator<U>&) const noexcept { return false; }
};

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (n > size_) {
            const bool growing = size_ != 0;
            const size_t spare = segments_.empty() ? 0 : segments_.back().capacity() - segments_.back().size();
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                segments_.back().resize(segments_.back().size() + extend, value);
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                Segment segment;
                segment.reserve(segment_capacity(count, previous, growing));
                segment.resize(count, value);
                segments_.push_back(std::move(segment));
                size_ += count;
                progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) segments_.back().resize(segments_.back().size() - (size_ - n));
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    using Segment = std::vector<uint8_t, SegmentAllocator<uint8_t>>;
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    static size_t segment_capacity(size_t count, size_t previous, bool growing) {
        if (!growing) return count;
        return std::max(count, std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

} // namespace strata::core
