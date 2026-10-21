// CPU-only tests for the disk tier of the parked conversations (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed, int32_t first = 1000) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(first + int32_t(i));
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

// one stage's image: `tokens` live tokens from `first`, checkpoints at a half and a quarter of them, two K/V layers
SavedConversation image(size_t tokens, uint8_t seed, int32_t first = 1000, int64_t lo = 0, int64_t hi = 48) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo; s.layer_hi = hi;
    s.cvec = false;
    s.live = checkpoint(tokens, seed, first);
    s.checkpoints.push_back(checkpoint(tokens / 2, uint8_t(seed + 9), first));    // the deepest first
    s.checkpoints.push_back(checkpoint(tokens / 4, uint8_t(seed + 19), first));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256; kv.page_size = 64; kv.pooled_rows = 9;
        kv.idx_dim = 128;
        kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(seed + layer));   // > 16 MiB: several segments
        kv.v = pattern(4096, uint8_t(seed + 10));
        kv.k_scale = pattern(512, uint8_t(seed + 20));
        kv.v_scale = pattern(512, uint8_t(seed + 30));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(seed + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// a layer split's image: the first stage's with one later stage's
SavedConversation split_image(size_t tokens, uint8_t seed, int32_t first = 1000) {
    SavedConversation s = image(tokens, seed, first, 0, 26);
    s.stage_images.push_back(image(tokens, uint8_t(seed + 50), first, 26, 48));
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size() ||
        a.stage_images.size() != b.stage_images.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    for (size_t i = 0; i < a.stage_images.size(); ++i)
        if (!same(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

std::vector<int32_t> prompt_of(size_t tokens, int32_t first = 1000) {
    std::vector<int32_t> p;
    for (size_t i = 0; i < tokens; ++i) p.push_back(first + int32_t(i));
    return p;
}

size_t tier_files(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().find(ConversationDiskTier::file_prefix()) != std::string::npos) ++n;
    return n;
}

void touch(const fs::path& p) { std::ofstream(p, std::ios::binary) << "x"; }

std::vector<std::string> lines;   // the tier's diagnostic lines
std::mutex lines_m;
ConversationDiskOptions opts(const fs::path& dir, uint64_t budget) {
    ConversationDiskOptions o;
    o.dir = dir.string();
    o.budget_bytes = budget;
    o.log = [](const std::string& l) { std::lock_guard<std::mutex> g(lines_m); lines.push_back(l); };
    return o;
}
bool logged(const char* what) {
    std::lock_guard<std::mutex> g(lines_m);
    for (const auto& l : lines) if (l.find(what) != std::string::npos) return true;
    return false;
}
size_t count_logged(const char* what) {
    std::lock_guard<std::mutex> g(lines_m);
    size_t n = 0;
    for (const auto& l : lines) n += l.find(what) != std::string::npos;
    return n;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata_conversation_disk_test_" + std::to_string(std::rand()));
    fs::remove_all(dir);
    const uint64_t GiB = 1ull << 30;
    std::string e;

    {   // off without a folder or a budget
        ConversationDiskTier off(opts("", GiB));
        check(off.open(e) && !off.enabled() && !off.put(image(10, 1)), "no folder: off, not an error");
        ConversationDiskTier off2(opts(dir, 0));
        check(off2.open(e) && !off2.enabled(), "no budget: off");
    }

    {   // start: this tier's leftovers (and their temporary files) go, other files stay
        fs::create_directories(dir);
        touch(dir / (std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin"));
        touch(dir / ("." + std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin.abcd.tmp"));
        touch(dir / "keep.bin");
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.enabled(), "open");
        check(tier_files(dir) == 0 && logged("2 stale files removed"), "open removes the tier's old and temporary files");
        check(fs::exists(dir / "keep.bin"), "open keeps other files");
        fs::remove(dir / "keep.bin");
    }

    {   // background write, match, read back exactly (a layer split: two files; every checkpoint)
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e), "open 2");
        const SavedConversation expect = split_image(600, 1);
        check(tier.put(split_image(600, 1)), "put a split image");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 2 && !tier.pending(), "written: one file per stage");
        check(logged("wrote 600 tokens") && logged("in 2 files"), "the write is logged");
        const auto live = tier.best(prompt_of(700), {}, false);
        check(live.tokens == 600 && live.live && live.id != 0, "best: the live state as a prefix");
        const auto ck = tier.best([] { auto p = prompt_of(300); p.push_back(7); return p; }(), {}, false);
        check(ck.tokens == 300 && !ck.live, "best: the deepest checkpoint as a prefix");
        check(tier.best([] { auto p = prompt_of(150); p.push_back(7); return p; }(), {}, false).tokens == 150,
              "best: an earlier checkpoint is kept too");
        const auto whole = tier.best(prompt_of(600), {}, false);
        check(whole.tokens == 300 && !whole.live, "best: the whole prompt held is no prefix (its last token is read)");
        check(tier.best(prompt_of(700), {}, true).tokens == 0, "best: another control-vector state does not match");
        check(tier.best(prompt_of(700, 5), {}, false).tokens == 0, "best: other tokens do not match");
        SavedConversation back;
        check(tier.take(live.id, live.tokens, back, e), "take");
        check(same(back, expect), "read back exactly: both stages, the live state and every checkpoint");
        check(tier.size() == 0 && tier.bytes() == 0 && tier_files(dir) == 0, "take removes the entry and its files");
        const auto st = tier.stats();
        check(st.queued == 1 && st.written == 1 && st.disk_hits == 1 && st.ram_hits == 0 && st.tokens_back == 600 &&
              st.bytes_read == st.bytes_written && st.bytes_written > (34u << 20), "counters");
        check(logged("disk hit: 600 tokens") && logged("totals:"), "the hit and the totals are logged");
    }

    {   // a queued image comes back from RAM: the writer is held at its first write
        std::promise<void> release;
        std::shared_future<void> gate = release.get_future().share();
        auto o = opts(dir, GiB);
        o.before_write = [gate] { gate.wait(); };
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e), "open 3");
        check(tier.put(image(400, 2)), "put a (the writer takes it and waits)");
        check(tier.put(image(400, 3, 5000)), "put b (queued behind a)");
        check(tier.pending(), "pending while the writer waits");
        const auto b = tier.best(prompt_of(450, 5000), {}, false);
        check(b.tokens == 400, "b matches in the queue");
        SavedConversation back;
        check(tier.take(b.id, b.tokens, back, e) && same(back, image(400, 3, 5000)), "b back from RAM");
        check(tier.stats().ram_hits == 1 && logged("RAM hit: 400 tokens"), "a RAM hit, logged");
        release.set_value();
        tier.wait_idle();
        check(!tier.pending() && tier.size() == 1 && tier_files(dir) == 1, "a written after the release");
        check(tier.best(prompt_of(450, 5000), {}, false).tokens == 0, "b is gone from the tier");
    }

    {   // xrig fix #3: a write that throws on the writer thread (here the test hook throws std::bad_alloc, as RAM
        // pressure would) drops that job only: no std::terminate, its waiters woken, and the next write goes through
        int calls = 0;   // the writer's thread only (wait_idle orders it)
        auto o = opts(dir, GiB);
        o.before_write = [&calls] { if (++calls == 1) throw std::bad_alloc(); };
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e) && tier.put(image(400, 34)), "throwing write: queued");
        tier.wait_idle();
        check(!tier.pending() && tier.size() == 0 && tier_files(dir) == 0 && tier.stats().write_failed == 1 &&
              logged("the write threw"), "throwing write: dropped, counted and logged; the queue is not blocked");
        check(tier.best(prompt_of(450), {}, false).tokens == 0, "throwing write: nothing matches it");
        check(tier.put(image(400, 35, 7000)), "throwing write: the next one is queued");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().written == 1 && tier_files(dir) == 1, "throwing write: the writer runs on");
    }

    {   // #879: clear() takes the conversations on the disk, their files too
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(image(400, 40)), "clear: put a");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 1, "clear: a on the disk");
        check(tier.clear() == 1 && tier.size() == 0 && tier_files(dir) == 0 && logged("removed all 1 conversations"),
              "clear: a and its file gone");
        check(tier.best(prompt_of(450), {}, false).tokens == 0, "clear: nothing matches");
    }

    {   // the budget: the oldest goes for a new one; an image above the budget is refused
        ConversationDiskTier tier(opts(dir, 25ull << 20));   // one ~17 MiB image, not two
        check(tier.open(e), "open 4");
        check(tier.put(image(400, 4)), "first queued");
        tier.wait_idle();
        check(tier.size() == 1, "first fits");
        check(tier.put(image(500, 5, 5000)), "second queued");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().evicted == 1, "second evicts the first");
        check(tier.best(prompt_of(450), {}, false).tokens == 0 && tier.best(prompt_of(600, 5000), {}, false).tokens == 500,
              "the newer one stays");
        ConversationDiskTier small(opts(dir, 1ull << 20));
        check(small.open(e) && !small.put(image(400, 6)) && small.size() == 0, "above the budget: refused");
    }
    check(tier_files(dir) == 0, "the tier removes its files when it ends");

    {   // xrig fix #4: a read the RAM preflight refuses: nothing restored, but the entry and its files stay (put back
        // as the newest), and a later read with room gives the conversation back exactly
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(split_image(400, 10)), "put for a refused read");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        bool kept = false;
        const bool ok = tier.take(m.id, m.tokens, back, e,
                                  [](uint64_t, std::string& why) { why = "no RAM"; return false; }, nullptr, &kept);
        check(!ok && kept && back.kv.empty() && back.stage_images.empty(), "refused read: empty image, kept");
        check(tier.size() == 1 && tier.bytes() > 0 && tier_files(dir) == 2 && logged("kept on the disk"),
              "refused read: the entry and its files stay");
        check(tier.stats().read_failed == 1, "a failed read is counted");
        const auto again = tier.best(prompt_of(450), {}, false);
        check(again.id == m.id && again.tokens == 400, "refused read: it still matches");
        check(tier.take(again.id, again.tokens, back, e, {}, nullptr, &kept) && !kept && same(back, split_image(400, 10)),
              "refused read: a later read gives it back exactly");
        check(tier.size() == 0 && tier.bytes() == 0 && tier_files(dir) == 0, "refused read: then the entry and files go");
    }

    {   // xrig fix #4: #879's clear() during a read (here from inside the RAM check, which runs without the lock) wins:
        // the refused read does not put its entry back, and its files go
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(image(400, 12)), "clear during a read: put");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        bool kept = true;
        auto admit = [&](uint64_t, std::string& why) { tier.clear(); why = "no RAM"; return false; };
        check(!tier.take(m.id, m.tokens, back, e, admit, nullptr, &kept) && !kept, "clear during a read: not kept");
        check(tier.size() == 0 && tier.bytes() == 0 && tier_files(dir) == 0, "clear during a read: entry and files gone");
    }

    {   // xrig fix #4: what the budget let in while a refused read ran goes first when its entry comes back (the one
        // just asked for is the newest)
        ConversationDiskTier tier(opts(dir, 25ull << 20));   // one ~17 MiB image, not two
        check(tier.open(e) && tier.put(image(400, 13)), "budget during a read: put a");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        bool kept = false;
        auto admit = [&](uint64_t, std::string& why) {
            check(tier.put(image(400, 14, 7000)), "budget during a read: put b");   // a is out of the list: no eviction
            tier.wait_idle();
            why = "no RAM";
            return false;
        };
        check(!tier.take(m.id, m.tokens, back, e, admit, nullptr, &kept) && kept, "budget during a read: a kept");
        check(tier.size() == 1 && tier_files(dir) == 1 && tier.bytes() <= (25ull << 20) &&
              tier.best(prompt_of(450), {}, false).tokens == 400 && tier.best(prompt_of(450, 7000), {}, false).tokens == 0,
              "budget during a read: b went for the budget, a stays");
    }

    {   // put_streamed: the K/V read from where it lives (here: the image's own buffers) gives the same files
        auto sources_of = [](const SavedConversation& img) {
            std::vector<SessionKvSource> out;
            for (const auto& kv : img.kv) {
                SessionKvSource s;
                s.format = kv.format; s.cells = kv.cells; s.heads = kv.heads; s.head_dim = kv.head_dim;
                s.page_size = kv.page_size; s.pooled_rows = kv.pooled_rows; s.idx_dim = kv.idx_dim;
                const ConversationBuffer* parts[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
                for (size_t p = 0; p < 5; ++p) s.sizes[p] = parts[p]->size();
                s.read = [parts](size_t part, size_t offset, void* dst, size_t n) {
                    return parts[part]->read(dst, offset, n);
                };
                out.push_back(std::move(s));
            }
            return out;
        };
        const SavedConversation expect = split_image(500, 30, 9000);
        std::vector<std::vector<SessionKvSource>> sources = {sources_of(expect), sources_of(expect.stage_images[0])};
        SavedConversation meta = split_image(500, 30, 9000);
        meta.kv.clear();
        meta.stage_images[0].kv.clear();
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put_streamed(meta, sources), "put_streamed: written");
        check(meta.stage_images.size() == 1 && meta.checkpoints.size() == 2, "put_streamed: the image is given back");
        check(tier.size() == 1 && tier_files(dir) == 2 && logged("streamed from the GPU"), "put_streamed: two files, logged");
        const auto m = tier.best(prompt_of(600, 9000), {}, false);
        SavedConversation back;
        check(m.tokens == 500 && tier.take(m.id, m.tokens, back, e) && same(back, expect),
              "put_streamed: read back exactly the image the sources came from");
        // xrig fix #3: a K/V source that throws while the second stage's file is written: false instead of the
        // exception, the first stage's file (and the partial one) removed, counted, and `meta` still given back
        std::vector<std::vector<SessionKvSource>> throwing = sources;
        throwing[1][0].read = [](size_t, size_t, void*, size_t) -> bool { throw std::runtime_error("injected"); };
        const size_t failed_before = tier.stats().write_failed;
        check(!tier.put_streamed(meta, throwing), "put_streamed: a throwing source is a failed write");
        check(meta.stage_images.size() == 1 && meta.checkpoints.size() == 2, "put_streamed: given back after a throw");
        check(tier.size() == 0 && tier_files(dir) == 0 && tier.stats().write_failed == failed_before + 1 &&
              logged("threw: injected"), "put_streamed: no file left after a throw; counted and logged");
    }

    {   // take with K/V targets: the files' K/V streamed into them (here: CPU buffers), the rest in `out`
        using Bytes = std::vector<uint8_t>;
        auto targets_for = [](const SavedConversation& img, std::vector<std::array<Bytes, 5>>& into) {
            into.assign(img.kv.size(), {});
            std::vector<SessionKvTarget> out;
            for (size_t j = 0; j < img.kv.size(); ++j) {
                const auto& kv = img.kv[j];
                SessionKvTarget t;
                t.format = kv.format; t.cells = kv.cells; t.heads = kv.heads; t.head_dim = kv.head_dim;
                t.page_size = kv.page_size; t.pooled_rows = kv.pooled_rows; t.idx_dim = kv.idx_dim;
                const ConversationBuffer* parts[5] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
                for (size_t p = 0; p < 5; ++p) { t.sizes[p] = parts[p]->size(); into[j][p].assign(t.sizes[p], 0); }
                std::array<Bytes, 5>* layer = &into[j];
                t.write = [layer](size_t part, size_t offset, const void* src, size_t n) {
                    std::memcpy((*layer)[part].data() + offset, src, n);
                    return true;
                };
                out.push_back(std::move(t));
            }
            return out;
        };
        auto bytes_of_buffer = [](const ConversationBuffer& b) {
            Bytes v(b.size());
            b.read(v.data(), 0, v.size());
            return v;
        };
        const SavedConversation expect = split_image(500, 31, 11000);
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(split_image(500, 31, 11000)), "streamed take: put");
        tier.wait_idle();
        std::vector<std::array<Bytes, 5>> got0, got1;
        std::vector<std::vector<SessionKvTarget>> kv = {targets_for(expect, got0), targets_for(expect.stage_images[0], got1)};
        const auto m = tier.best(prompt_of(600, 11000), {}, false);
        SavedConversation meta;
        bool kept = false;   // xrig fix #4: a streamed read the RAM check refuses keeps the entry too
        check(!tier.take(m.id, m.tokens, meta, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; }, &kv,
                         &kept) && kept && tier.size() == 1 && tier_files(dir) == 2,
              "streamed take: a refused RAM check keeps the entry and its files");
        check(m.tokens == 500 && !m.queued && m.bytes > 0 && tier.take(m.id, m.tokens, meta, e, {}, &kv),
              "streamed take: read");
        bool same_kv = true;
        for (size_t j = 0; j < expect.kv.size(); ++j) {
            const ConversationBuffer* parts[5] = {&expect.kv[j].k, &expect.kv[j].v, &expect.kv[j].k_scale,
                                                  &expect.kv[j].v_scale, &expect.kv[j].pooled};
            for (size_t p = 0; p < 5; ++p) same_kv = same_kv && got0[j][p] == bytes_of_buffer(*parts[p]);
        }
        check(same_kv && logged("streamed into the GPU"), "streamed take: the targets hold the K/V exactly");
        check(meta.kv.size() == expect.kv.size() && meta.kv[0].k.size() == 0 &&
              same_checkpoint(meta.live, expect.live) && meta.checkpoints.size() == expect.checkpoints.size() &&
              meta.stage_images.size() == 1, "streamed take: the running state and checkpoints in RAM, no K/V");
        check(tier.put(image(400, 32, 12000)), "streamed take: put a second");
        tier.wait_idle();
        const SavedConversation other = image(400, 32, 12000);
        std::vector<std::array<Bytes, 5>> bad;
        std::vector<std::vector<SessionKvTarget>> wrong = {targets_for(other, bad)};
        wrong[0][0].sizes[0] += 1;
        const auto m2 = tier.best(prompt_of(450, 12000), {}, false);
        check(!tier.take(m2.id, m2.tokens, meta, e, {}, &wrong, &kept) && e.find("size differs") != std::string::npos,
              "streamed take: a K/V part of another size is refused");
        check(!kept && tier.size() == 0 && tier_files(dir) == 0, "streamed take: a refused file is not kept (xrig fix #4)");
    }

    {   // the RAM cache's evictions go to the disk tier (set_on_evict)
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e), "open 6");
        ConversationCache cache(GiB, 1);
        cache.set_on_evict([&](SavedConversation&& s) { tier.put(std::move(s)); });
        check(cache.put(image(400, 11)), "park a");
        check(cache.put(image(400, 12, 8000)), "park b: a leaves RAM");
        tier.wait_idle();
        check(cache.size() == 1 && cache.evictions() == 1 && tier.size() == 1, "a is on the disk");
        check(tier.best(prompt_of(450), {}, false).tokens == 400, "a matches on the disk");
        check(cache.best(prompt_of(450, 8000), {}, false).tokens == 400, "b stays in RAM");
    }

    {   // persist: the files outlive the tier, and the next tier with the same identity takes them in
        auto persistent = [&](uint64_t model) {
            auto o = opts(dir, GiB);
            o.persist = true;
            o.identity = {model, 7};
            return o;
        };
        const std::string prefix = ConversationDiskTier::file_prefix();
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.persistent(), "persist: open");
            check(tier.put(split_image(500, 20)) && tier.put(image(400, 21, 6000)), "persist: two queued");
        }   // the end waits for the queue and keeps the files
        check(tier_files(dir) == 5 && logged("2 conversations kept on the disk"), "persist: 3 images + 2 indexes kept");
        touch(dir / ("." + prefix + "x-1-0.bin.abcd.tmp"));
        touch(dir / (prefix + "orphan-9-0.bin"));
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.size() == 2 && tier_files(dir) == 5, "persist: taken in, leftovers removed");
            check(logged("2 conversations from earlier runs"), "persist: the start is logged");
            const auto m = tier.best(prompt_of(600), {}, false);
            check(m.tokens == 500 && m.live, "persist: matches after a restart");
            SavedConversation back;
            bool kept = false;   // xrig fix #4: a refused RAM check keeps the index too
            check(!tier.take(m.id, m.tokens, back, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; },
                             nullptr, &kept) && kept && tier.size() == 2 && tier_files(dir) == 5,
                  "persist: a refused read keeps the images and the index");
            const bool read = tier.take(m.id, m.tokens, back, e);
            if (!read) std::fprintf(stderr, "take: %s\n", e.c_str());
            check(read, "persist: read back after a restart");
            check(same(back, split_image(500, 20)), "persist: the image is exactly the one written");
            check(tier_files(dir) == 2, "persist: its images and index are gone");
        }
        {
            ConversationDiskTier other(persistent(2));   // another model, engine version or settings
            check(other.open(e) && other.size() == 0 && tier_files(dir) == 0 && logged("1 refused"),
                  "persist: another identity's files are refused and removed");
        }
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.put(image(300, 22)), "persist: put for a damaged index");
        }
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".idx") fs::resize_file(f.path(), 20);
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.size() == 0 && tier_files(dir) == 0, "persist: a damaged index is refused");
        }
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.put(image(400, 23)) && tier.put(image(400, 24, 6000)),
                  "persist: put a and b, the same length");
        }
        std::vector<fs::path> bins;   // a's image first: the names differ only in the id
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".bin") bins.push_back(f.path());
        std::sort(bins.begin(), bins.end());
        check(bins.size() == 2, "persist: two images");
        fs::copy_file(bins[0], bins[1], fs::copy_options::overwrite_existing);   // b's index now names a's image
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.size() == 2, "persist: both taken in");
            const auto m = tier.best(prompt_of(450, 6000), {}, false);
            check(m.tokens == 400, "persist: b matches by its index");
            SavedConversation back;
            bool kept = true;
            check(!tier.take(m.id, m.tokens, back, e, {}, nullptr, &kept) && back.kv.empty() &&
                  logged("hold another conversation"), "persist: a's image under b's index is refused, never restored as b");
            check(!kept && tier.size() == 1, "persist: the mismatched entry is not kept (xrig fix #4)");
            check(tier.best(prompt_of(450), {}, false).tokens == 400, "persist: a is still there");
        }
    }

    {   // xrig hardening: an index naming a file outside the folder (absolute or "..") is refused, and that file stays
        auto persistent5 = [&] {
            auto o = opts(dir, GiB);
            o.persist = true;
            o.identity = {5, 7};
            return o;
        };
        const fs::path outside = dir.parent_path() / "strata-conv-outside-sentinel-0.bin";
        touch(outside);
        for (const std::string& evil : {outside.string(), std::string("../") + outside.filename().string()}) {
            {
                ConversationDiskTier tier(persistent5());
                check(tier.open(e) && tier.put(image(300, 25)), "hardening: put");
            }
            for (const auto& f : fs::directory_iterator(dir)) {
                if (f.path().extension() != ".idx") continue;
                std::ifstream in(f.path(), std::ios::binary);
                std::string b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                in.close();
                const size_t at = 8 * 3 + 8 + 1 + 8 + 4;   // magic, model, config, time, cvec, bytes, file count
                uint32_t n = 0;
                std::memcpy(&n, b.data() + at, 4);
                const uint32_t m = (uint32_t) evil.size();
                b = b.substr(0, at) + std::string(reinterpret_cast<const char*>(&m), 4) + evil + b.substr(at + 4 + n);
                std::ofstream out(f.path(), std::ios::binary | std::ios::trunc);
                out.write(b.data(), (std::streamsize) b.size());
            }
            {
                ConversationDiskTier tier(persistent5());
                check(tier.open(e) && tier.size() == 0, "hardening: an index naming a path is refused");
                tier.clear();
            }
            check(fs::exists(outside), "hardening: the file outside the folder is never deleted");
        }
        fs::remove(outside);
    }

    {   // xrig fix #5: the index is created private (0600, as the images are) under a random temporary name that
        // does not stay behind; an oversized index or a link at an index's name is refused at start, and a link's
        // target outside the folder is neither read nor deleted
        auto persistent6 = [&] {
            auto o = opts(dir, GiB);
            o.persist = true;
            o.identity = {6, 7};
            return o;
        };
        {
            ConversationDiskTier tier(persistent6());
            check(tier.open(e) && tier.put(image(300, 26)), "fix #5: put");
        }
        fs::path idx;
        size_t temps = 0;
        for (const auto& f : fs::directory_iterator(dir)) {
            if (f.path().extension() == ".idx") idx = f.path();
            if (f.path().extension() == ".tmp") ++temps;
        }
        check(!idx.empty() && temps == 0, "fix #5: one index, no temporary file left");
#ifndef _WIN32
        const auto other = fs::perms::group_all | fs::perms::others_all;
        check((fs::status(idx).permissions() & other) == fs::perms::none, "fix #5: the index is private (0600)");
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".bin")
                check((fs::status(f.path()).permissions() & other) == fs::perms::none, "fix #5: the image is private");
        check((fs::status(dir).permissions() & other) == fs::perms::none, "fix #5: the folder is private (0700)");
        // a link at the index's name, to the real index moved outside the folder: refused, the target stays
        const fs::path moved = dir.parent_path() / ("strata_conversation_disk_test_idx_" + std::to_string(std::rand()));
        fs::rename(idx, moved);
        fs::create_symlink(moved, idx);
        {
            ConversationDiskTier tier(persistent6());
            check(tier.open(e) && tier.size() == 0 && tier_files(dir) == 0, "fix #5: a linked index is refused");
        }
        check(fs::exists(moved), "fix #5: the link's target is not deleted");
        fs::remove(moved);
#endif
        // an index over the size cap (sparse: nothing is written) is refused without being read, and removed
        const fs::path big = dir / (std::string(ConversationDiskTier::file_prefix()) + "0123-1.idx");
        touch(big);
        fs::resize_file(big, (64ull << 20) + 1);
        {
            ConversationDiskTier tier(persistent6());
            check(tier.open(e) && tier.size() == 0 && !fs::exists(big),
                  "fix #5: an index over 64 MiB is refused and removed");
        }
    }

    {   // xrig tier fix: wait_idle_for gives up after its bound and beats meanwhile; the write still ends later
        std::promise<void> release;
        std::shared_future<void> gate = release.get_future().share();
        auto o = opts(dir, GiB);
        o.before_write = [gate] { gate.wait(); };
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e) && tier.put(image(400, 50)), "bounded wait: a write held");
        int beats = 0;
        const bool idle = tier.wait_idle_for(600, [&beats] { ++beats; });
        check(!idle && beats >= 2 && tier.pending(), "bounded wait: gives up after the bound, beating meanwhile");
        check(tier.stats().wait_timeouts == 1 && logged("gave up waiting"), "bounded wait: counted and logged");
        release.set_value();
        check(tier.wait_idle_for(-1, nullptr) && !tier.pending() && tier.size() == 1,
              "bounded wait: no limit (-1) waits like wait_idle; the held write ended");
        check(tier.wait_idle_for(10, nullptr), "bounded wait: an idle tier returns at once");
        tier.clear();
    }

    {   // xrig tier fix: the write gate - short conversations are not written
        auto o = opts(dir, GiB);
        o.min_write_tokens = 500;
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e) && !tier.put(image(400, 51)), "min tokens: a 400-token conversation is not written");
        check(tier.put(image(600, 52, 9000)), "min tokens: a 600-token one is");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().skipped_short == 1 && logged("under the minimum tokens"),
              "min tokens: counted and logged");
        tier.clear();
    }

    {   // xrig tier fix: the same state already on the disk (live or a checkpoint) is not written again; a longer
        // entry alone does not count (conversation_prefix needs the stored tokens to be a prefix of the prompt)
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(image(400, 53)), "same state: the first copy is written");
        tier.wait_idle();
        check(!tier.put(image(400, 54)) && tier.stats().skipped_covered == 1 && logged("the disk already holds it"),
              "same state: the same 400 tokens again are not written");
        check(!tier.put(image(200, 55)), "same state: a state equal to the entry's checkpoint is not written");
        check(tier.put(image(300, 56)), "same state: a shorter state that is no checkpoint of it IS written");
        check(tier.put(image(800, 57)), "same state: a longer one is written");
        tier.wait_idle();
        check(tier.size() == 3, "same state: three entries on the disk");
        tier.clear();
    }

    {   // xrig tier fix: the daily write budget
        const uint64_t one = image(400, 58).bytes(), small = image(100, 250, 9000).bytes();
        check(small < one, "day budget: a 100-token image is smaller than a 400-token one");
        auto o = opts(dir, GiB);
        o.day_write_bytes = one * 2 + small;   // room for two, and then only for the small one
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e) && tier.put(image(400, 58, 1000)) && tier.put(image(400, 59, 3000)),
              "day budget: two writes fit");
        check(!tier.put(image(400, 60, 5000)) && tier.stats().skipped_day == 1 && logged("over the daily write budget"),
              "day budget: the third is not written, counted and logged");
        tier.wait_idle();
        check(tier.size() == 2, "day budget: two entries on the disk");
        check(count_logged("daily write budget reached") == 1 && logged("at the earliest "),
              "day budget: one notice that the budget is reached, with the earliest time");
        const size_t refused_lines = count_logged("over the daily write budget");
        for (int i = 0; i < 150; ++i) tier.put(image(400, uint8_t(61 + i), 7000 + 2 * i));
        check(tier.stats().skipped_day == 151 && count_logged("over the daily write budget") == refused_lines + 1 &&
                  count_logged("daily write budget reached") == 1,
              "day budget: 150 more refusals log one line (1 in 100) and no second notice");
        check(tier.put(image(100, 250, 9000)) && logged("daily write budget has room again") &&
                  logged("resumes after 151 refused writes"),
              "day budget: a write that fits again is written, with a notice");
        tier.wait_idle();
        tier.clear();
    }

    fs::remove_all(dir);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
