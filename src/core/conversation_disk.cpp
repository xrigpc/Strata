// src/core/conversation_disk.cpp - the disk tier of the parked conversations.  See the header.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <system_error>
#include <utility>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;

// The second half of every file's identity: with this run's random number as the first, a file another run (or
// another tier) wrote is refused on read.
constexpr uint64_t kTierConfig = 0x5354524154414449ull;   // "STRATADI"

// ids and images only: the part of a checkpoint best() matches against
ConversationCheckpoint key_of(const ConversationCheckpoint& c) {
    ConversationCheckpoint k;
    k.ids = c.ids;
    k.imgs = c.imgs;
    return k;
}

// The payload is the conversation its index names: the same keys and control-vector state, in every stage.
bool holds(const SavedConversation& s, const ConversationCheckpoint& live,
           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    auto same = [](const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
        return x.ids == y.ids && x.imgs == y.imgs;
    };
    if (s.cvec != cvec || !same(s.live, live) || s.checkpoints.size() != checkpoints.size()) return false;
    for (size_t i = 0; i < checkpoints.size(); ++i)
        if (!same(s.checkpoints[i], checkpoints[i])) return false;
    for (const auto& stage : s.stage_images)
        if (!holds(stage, live, checkpoints, cvec)) return false;
    return true;
}

// a file of this tier (also a temporary one session_file_write left behind: ".<name>.<hex>.tmp")
bool tier_file(const std::string& name) {
    const std::string prefix = ConversationDiskTier::file_prefix();
    return name.rfind(prefix, 0) == 0 || name.rfind("." + prefix, 0) == 0;
}

// a data file name this tier writes itself ("strata-conv-<hex>-<id>-<k>.bin"): no separator, no "..", nothing else
bool safe_bin_name(const std::string& name) {
    const std::string prefix = ConversationDiskTier::file_prefix();
    if (name.size() <= prefix.size() + 4 || name.size() > 200 || name.rfind(prefix, 0) != 0 ||
        name.compare(name.size() - 4, 4, ".bin") != 0)
        return false;
    for (char c : name)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || c == '-' || c == '.')) return false;
    return name.find("..") == std::string::npos;
}

// xrig fix #3: never throws (the path is a copy: std::bad_alloc), as it runs on the failure paths, where a second
// exception would end the engine
void remove_file(const std::string& path) noexcept {
    try {
        std::error_code ec;
        fs::remove(fs::path(path), ec);
    } catch (...) {
    }
}

// xrig fix #3: what a caught exception says, copied into a fixed buffer (no allocation: the usual one here is
// std::bad_alloc, and the exception object is gone once its handler ends)
template<size_t N> void describe(char (&buf)[N], const std::exception* x) noexcept {
    std::snprintf(buf, N, "%s", x ? x->what() : "an unknown exception");
}

// The index file of a persistent entry: native-endian fields (the session files refuse a big-endian build too),
// counts before arrays.
constexpr uint64_t kIndexMagic = 0x3158444943525453ull;   // "STRCIDX1"
// xrig fix #5: the largest index read_index() reads: 16M token ids over the live state and its checkpoints (333K
// tokens with 50 checkpoints), far past what the engine writes.  A bigger one is refused (removed at start).
constexpr uint64_t kMaxIndexBytes = 64ull << 20;

// xrig fix #5: a small file read whole, only when it is a regular file (never what a symbolic link at its name points
// to: O_NOFOLLOW; O_NONBLOCK so a FIFO cannot hang the start) of at most `max` bytes - the size is checked on the
// opened file before anything is allocated, so a huge or endless file in the folder costs nothing.
bool read_small_file(const std::string& path, uint64_t max, std::string& out) {
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st {};
    bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 && (uint64_t) st.st_size <= max;
    if (ok) {
        out.resize((size_t) st.st_size);
        for (size_t at = 0; ok && at < out.size();) {
            const ssize_t r = ::read(fd, &out[at], out.size() - at);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) ok = false;
            else at += (size_t) r;
        }
    }
    ::close(fd);
    return ok;
#else
    std::error_code ec;
    const fs::path p(path);
    if (!fs::is_regular_file(fs::symlink_status(p, ec)) || ec) return false;
    const uint64_t size = fs::file_size(p, ec);
    if (ec || size > max) return false;
    std::ifstream f(p, std::ios::binary);
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return out.size() == size;
#endif
}

struct IndexWriter {
    std::string b;
    template<class T> void put(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void str(const std::string& s) { put<uint32_t>((uint32_t) s.size()); b += s; }
    void key(const ConversationCheckpoint& c) {
        put<uint32_t>((uint32_t) c.ids.size());
        b.append(reinterpret_cast<const char*>(c.ids.data()), c.ids.size() * sizeof(int32_t));
        put<uint32_t>((uint32_t) c.imgs.size());
        for (const auto& i : c.imgs) { put<int64_t>(i.start); put<uint64_t>(i.hash); }
    }
};

struct IndexReader {
    const std::string& b;
    size_t at = 0;
    bool ok = true;
    size_t left() const { return b.size() - at; }
    template<class T> T get() {
        T v{};
        if (left() < sizeof v) { ok = false; return v; }
        std::memcpy(&v, b.data() + at, sizeof v);
        at += sizeof v;
        return v;
    }
    std::string str() {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() < n) { ok = false; return {}; }
        at += n;
        return b.substr(at - n, n);
    }
    bool key(ConversationCheckpoint& c) {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() / sizeof(int32_t) < n) return ok = false;
        c.ids.resize(n);
        std::memcpy(c.ids.data(), b.data() + at, n * sizeof(int32_t));
        at += n * sizeof(int32_t);
        const uint32_t m = get<uint32_t>();
        if (!ok || left() / 16 < m) return ok = false;
        for (uint32_t i = 0; i < m; ++i) c.imgs.push_back({get<int64_t>(), get<uint64_t>()});
        return ok;
    }
};

double mib(uint64_t bytes) { return double(bytes) / (1024.0 * 1024.0); }
double mbps(uint64_t bytes, double ms) { return ms > 0 ? double(bytes) / 1e6 / (ms / 1000.0) : 0.0; }
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

#if defined(__GNUC__)
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

}  // namespace

ConversationDiskTier::ConversationDiskTier(ConversationDiskOptions options) : options_(std::move(options)) {
    std::random_device rd;
    nonce_ = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
}

ConversationDiskTier::~ConversationDiskTier() {
    if (open_ && options_.persist) wait_idle();   // what is queued belongs to the next start
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
        dropped = queue_.size();
        queue_.clear();   // not written: dropped, as the RAM cache would have
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    std::lock_guard<std::mutex> lock(m_);
    if (!open_) return;
    if (options_.persist) {
        log(fmt("end: %zu conversations kept on the disk for the next start (%.0f MiB)", entries_.size(), mib(bytes_)));
        return;
    }
    if (dropped || !entries_.empty())
        log(fmt("end: %zu queued dropped, %zu conversations removed from the disk", dropped, entries_.size()));
    while (!entries_.empty()) remove_locked(0, true);
}

SessionFileIdentity ConversationDiskTier::file_identity() const {
    return options_.persist ? options_.identity : SessionFileIdentity{nonce_, kTierConfig};
}

bool ConversationDiskTier::write_index(Entry& e, std::string& error) const {
    char name[96];
    std::snprintf(name, sizeof name, "%s%016llx-%llu.idx", file_prefix(), (unsigned long long) nonce_,
                  (unsigned long long) e.id);
    const fs::path path = fs::path(options_.dir) / name;
    // xrig fix #5: the temporary name has a random part (it was a predictable ".<name>.tmp"), and it is created new
    // below (O_CREAT|O_EXCL|O_NOFOLLOW: never an existing file or a link someone put there) with mode 0600, as the
    // session files are.  A leftover still starts with "." + file_prefix(): open() removes it at the next start.
    char rnd[24];
    std::snprintf(rnd, sizeof rnd, "%016llx", (unsigned long long) (((uint64_t) std::random_device{}() << 32) ^
                                                                      std::random_device{}()));
    const fs::path temp = fs::path(options_.dir) / ("." + std::string(name) + "." + rnd + ".tmp");
    e.written_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const SessionFileIdentity id = file_identity();
    IndexWriter out;
    out.put(kIndexMagic);
    out.put(id.model);
    out.put(id.config);
    out.put<int64_t>(e.written_s);
    out.put<uint8_t>(e.cvec ? 1 : 0);
    out.put<uint64_t>(e.bytes);
    out.put<uint32_t>((uint32_t) e.files.size());
    for (const auto& f : e.files) out.str(fs::path(f).filename().string());
    out.key(e.live);
    out.put<uint32_t>((uint32_t) e.checkpoints.size());
    for (const auto& c : e.checkpoints) out.key(c);
#ifndef _WIN32
    {
        const std::string temp_s = temp.string();
        const int fd = ::open(temp_s.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd < 0) {
            const int err = errno;
            error = "cannot create the index " + temp_s + ": " + std::strerror(err);
            return false;
        }
        bool ok = true;
        int err = 0;
        for (size_t at = 0; ok && at < out.b.size();) {
            const ssize_t w = ::write(fd, out.b.data() + at, out.b.size() - at);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { err = w < 0 ? errno : 0; ok = false; }
            else at += (size_t) w;
        }
        if (::close(fd) != 0 && ok) { err = errno; ok = false; }
        if (!ok) {
            remove_file(temp_s);
            error = "cannot write the index " + temp_s + (err ? std::string(": ") + std::strerror(err) : std::string());
            return false;
        }
    }
#else
    {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        f.write(out.b.data(), (std::streamsize) out.b.size());
        if (!f) {
            error = "cannot write the index " + temp.string();
            remove_file(temp.string());
            return false;
        }
    }
#endif
    std::error_code ec;
    fs::rename(temp, path, ec);   // a reader sees the whole index or none
    if (ec) {
        error = "cannot publish the index: " + ec.message();
        remove_file(temp.string());
        return false;
    }
    e.index = path.string();
    return true;
}

bool ConversationDiskTier::read_index(const std::string& path, Entry& e) const {
    std::string b;
    if (!read_small_file(path, kMaxIndexBytes, b)) return false;   // xrig fix #5: bounded, no link, regular files only
    IndexReader in{b};
    const SessionFileIdentity id = file_identity();
    if (in.get<uint64_t>() != kIndexMagic || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config)
        return false;
    e.written_s = in.get<int64_t>();
    e.cvec = in.get<uint8_t>() != 0;
    e.bytes = in.get<uint64_t>();
    const uint32_t nfiles = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nfiles; ++i) {
        // xrig hardening: a name from the index is a bare file name of this tier in the folder - never a path (an
        // absolute name or ".." would make the budget and take() read and delete a file outside the folder)
        const std::string name = in.str();
        if (!in.ok || !safe_bin_name(name)) return false;
        e.files.push_back((fs::path(options_.dir) / name).string());
    }
    if (!in.key(e.live)) return false;
    const uint32_t nck = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nck; ++i) {
        ConversationCheckpoint c;
        if (in.key(c)) e.checkpoints.push_back(std::move(c));
    }
    if (!in.ok || in.left() != 0 || e.files.empty()) return false;
    std::error_code ec;
    for (const auto& file : e.files)
        if (!fs::is_regular_file(fs::path(file), ec)) return false;
    e.index = path;
    return true;
}

std::string ConversationDiskTier::scan_folder(size_t& stale, std::error_code& ec) {
    std::vector<Entry> found;
    std::vector<std::string> others;
    size_t refused = 0;
    for (fs::directory_iterator it(fs::path(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || !tier_file(it->path().filename().string())) continue;
        Entry e;
        const bool index = it->path().extension() == ".idx";
        const bool taken = index && read_index(it->path().string(), e);
        if (taken) found.push_back(std::move(e));
        else others.push_back(it->path().string());
        refused += index && !taken;
    }
    // the files of the entries taken in stay; the rest goes (refused indexes and their files, interrupted writes)
    std::set<std::string> kept;
    for (const auto& e : found) kept.insert(e.files.begin(), e.files.end());
    for (const auto& p : others)
        if (!kept.count(p)) { remove_file(p); ++stale; }
    std::sort(found.begin(), found.end(), [](const Entry& a, const Entry& b) { return a.written_s < b.written_s; });
    for (auto& e : found) {
        e.id = ++next_id_;
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
    }
    size_t over = 0;
    while (!entries_.empty() && bytes_ > options_.budget_bytes) {
        remove_locked(0, true);
        ++over;
    }
    return fmt("persistent (budget only): %zu conversations from earlier runs (%.0f MiB), %zu refused (another model, "
               "engine version or settings), %zu over the budget removed", entries_.size(), mib(bytes_), refused, over);
}

void ConversationDiskTier::log(const std::string& line) const {
    if (options_.log) options_.log(line);
    else std::fprintf(stderr, "strata serve: conversation disk: %s\n", line.c_str());
}

bool ConversationDiskTier::open(std::string& error) {
    open_ = false;
    if (options_.dir.empty() || options_.budget_bytes == 0) return true;   // off: nothing to open
    std::error_code ec;
    const fs::path dir(options_.dir);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        error = "cannot create the folder " + options_.dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);   // the files hold conversations
    if (ec) {
        // xrig fix #5: a folder that cannot be made private (not ours, a read-only mount, ...) is refused - the tier
        // stays off, with this line in the log - instead of holding conversations readable by others
        error = "cannot make the folder " + options_.dir + " private (chmod 0700): " + ec.message();
        return false;
    }
#endif
    size_t stale = 0;
    const std::string taken_in = scan_folder(stale, ec);
    if (ec) {
        error = "cannot list the folder " + options_.dir + ": " + ec.message();
        return false;
    }
    open_ = true;
    thread_ = std::thread([this] { writer(); });
    const std::string kept = options_.persist ? taken_in : "removed at the end";
    log(fmt("on: %s, budget %.0f MiB, %s; %zu stale files removed", options_.dir.c_str(), mib(options_.budget_bytes),
            kept.c_str(), stale));
    return true;
}

void ConversationDiskTier::remove_locked(size_t index, bool delete_files) {
    Entry& e = entries_.at(index);
    if (!e.index.empty()) remove_file(e.index);   // first, always: an entry gone from the list is not taken in again
    if (delete_files)
        for (const auto& f : e.files) remove_file(f);
    bytes_ -= e.bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

bool ConversationDiskTier::put(SavedConversation&& image) {
    if (!open_) return false;
    // xrig fix #3: the keys copy the token lists and the queue may need a new block; a throw (std::bad_alloc under RAM
    // pressure) drops the image like a refused one instead of escaping into the RAM cache's eviction on the request
    // thread
    char thrown[160] = "";
    try {
        return put_body(std::move(image));
    } catch (const std::exception& x) {
        describe(thrown, &x);
    } catch (...) {
        describe(thrown, nullptr);
    }
    {
        std::lock_guard<std::mutex> lock(m_);
        ++stats_.write_failed;
    }
    try { log(fmt("dropped a conversation: queueing it threw (%s)", thrown)); } catch (...) {}
    return false;
}

bool ConversationDiskTier::put_body(SavedConversation&& image) {
    Job job;
    job.image = std::move(image);
    job.entry.live = key_of(job.image.live);
    for (const auto& c : job.image.checkpoints) job.entry.checkpoints.push_back(key_of(c));
    job.entry.cvec = job.image.cvec;
    job.entry.bytes = job.image.bytes();
    const size_t tokens = job.entry.live.ids.size();
    if (job.entry.bytes > options_.budget_bytes) {
        log(fmt("dropped %zu tokens: %.0f MiB is over the disk budget", tokens, mib(job.entry.bytes)));
        std::lock_guard<std::mutex> lock(m_);
        ++stats_.write_failed;
        return false;
    }
    size_t depth = 0;
    uint64_t queued_bytes = 0;
    const uint64_t estimate = job.entry.bytes;
    const char* why = nullptr;   // xrig tier fix: the write gate's refusal, if any
    {
        std::lock_guard<std::mutex> lock(m_);
        why = write_gate(job.entry.live, job.entry.cvec, estimate);
        if (!why) {
            job.entry.id = ++next_id_;
            queue_.push_back(std::move(job));
            ++stats_.queued;   // xrig fix #3: once it is in the queue (a throw above counts as a failed write instead)
            depth = queue_.size();
            for (const auto& j : queue_) queued_bytes += j.entry.bytes;
        }
    }
    if (why) {   // not written: the caller drops it, as for a refused write (write_gate counted it)
        try { log(fmt("not written: %zu tokens, %.0f MiB (%s)", tokens, mib(estimate), why)); } catch (...) {}
        return false;
    }
    cv_.notify_all();
    try {   // xrig fix #3: queued: a failure to log is not a failed put
        log(fmt("queued %zu tokens (%.0f MiB in RAM until written; queue %zu, %.0f MiB)", tokens, mib(estimate), depth,
                mib(queued_bytes)));
    } catch (...) {
    }
    return true;
}

void ConversationDiskTier::writer() {
#if defined(__linux__)
    // below the serving threads: the write is background work, decode and prompt reading come first
    setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 10);
    // xrig tier fix: the lowest best-effort I/O class too, so the snapshot writes yield the disk to the model's own
    // reads (--ple-io direct reads the PLE table on the token path).  Best-effort, not idle: idle could starve the
    // queue, whose images hold RAM until written.  Ignored when the I/O scheduler does not use priorities.
    {
        constexpr int ioprio_class_be = 2, ioprio_class_shift = 13, ioprio_who_process = 1;
        syscall(SYS_ioprio_set, ioprio_who_process, (int) syscall(SYS_gettid), (ioprio_class_be << ioprio_class_shift) | 7);
    }
#endif
    for (;;) {
        Job job;
        // xrig fix #3: an exception that left this thread would end the engine (std::terminate).  store() catches its
        // own; this catches the rest (std::bad_alloc copying the keys below, the write's set-up, a test's hook).  The
        // job is then dropped like a failed write: counted, logged once, and writing_ cleared with its waiters woken,
        // so the queue, take() and wait_idle() never hang on it.
        char thrown[160] = "";
        size_t tokens = 0;
        try {
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (stop_) return;
                job = std::move(queue_.front());
                queue_.pop_front();
                tokens = job.entry.live.ids.size();
                // xrig fix #3: the copy (which can throw) before the flag: best() never scans half a copy
                writing_entry_ = job.entry;   // best() still sees it; take() waits for its file
                writing_ = true;
            }
            write(job);
        } catch (const std::exception& x) {
            describe(thrown, &x);
        } catch (...) {
            describe(thrown, nullptr);
        }
        { Job gone = std::move(job); }   // xrig fix #3: its RAM goes back (written or not) before the waiters wake
        {
            std::lock_guard<std::mutex> lock(m_);
            writing_ = false;
            writing_entry_ = Entry{};
            if (thrown[0]) ++stats_.write_failed;
        }
        cv_.notify_all();
        if (thrown[0]) {
            try { log(fmt("dropped %zu tokens: the write threw (%s)", tokens, thrown)); } catch (...) {}
        }
    }
}

void ConversationDiskTier::write(Job& job) {
    if (options_.before_write) options_.before_write();
    std::vector<SavedConversation> parts = std::move(job.image.stage_images);   // one file per image, none nested
    job.image.stage_images.clear();
    const SessionFileIdentity id = file_identity();
    store(std::move(job.entry), parts.size() + 1,
          [&](size_t k, const std::string& path, size_t& written, std::string& error) {
              return session_file_write(path, k == 0 ? job.image : parts[k - 1], id, written, error, write_options());
          }, "");
    job.image = SavedConversation{};   // the RAM goes back now, written or not
}

bool ConversationDiskTier::put_streamed(SavedConversation& meta,
                                        const std::vector<std::vector<SessionKvSource>>& sources) {
    if (!open_ || sources.size() != meta.stage_images.size() + 1) return false;
    Entry e;
    e.live = key_of(meta.live);
    for (const auto& c : meta.checkpoints) e.checkpoints.push_back(key_of(c));
    e.cvec = meta.cvec;
    e.bytes = meta.bytes();
    for (const auto& image : sources)
        for (const auto& layer : image)
            for (size_t part : layer.sizes) e.bytes += part;
    if (e.bytes > options_.budget_bytes) {
        log(fmt("dropped %zu tokens: %.0f MiB is over the disk budget", e.live.ids.size(), mib(e.bytes)));
        std::lock_guard<std::mutex> lock(m_);
        ++stats_.write_failed;
        return false;
    }
    {
        std::unique_lock<std::mutex> lock(m_);
        if (const char* why = write_gate(e.live, e.cvec, e.bytes)) {
            lock.unlock();
            log(fmt("not written: %zu tokens, %.0f MiB, streamed (%s)", e.live.ids.size(), mib(e.bytes), why));
            return false;
        }
        e.id = ++next_id_;
    }
    std::vector<SavedConversation> parts = std::move(meta.stage_images);
    meta.stage_images.clear();
    const SessionFileIdentity id = file_identity();
    const bool ok = store(std::move(e), parts.size() + 1,
                          [&](size_t k, const std::string& path, size_t& written, std::string& error) {
                              return session_file_write(path, k == 0 ? meta : parts[k - 1], sources[k], id, written,
                                                        error, write_options());
                          }, ", streamed from the GPU");
    meta.stage_images = std::move(parts);   // given back as it came: the caller lent its checkpoints
    return ok;
}

SessionWriteOptions ConversationDiskTier::write_options() const {
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = false;   // a cache: a file a power loss damages fails its hash check and is read again from the prompt
    return wo;
}

bool ConversationDiskTier::store(Entry e, size_t images, const WriteFile& write_file, const char* how) {
    const size_t tokens = e.live.ids.size();
    size_t made_room = 0;
    auto t0 = std::chrono::steady_clock::now();
    std::string error;
    bool ok = true;
    // xrig fix #3: anything in this block can throw - std::bad_alloc under RAM pressure is the likely one (a path, the
    // files list, the index, the entry list's next block, a K/V source).  On the writer thread an escaped exception
    // ends the engine (std::terminate); on the request thread it skips put_streamed's hand-back of `meta`.  So a throw
    // is a failed write like any other: its files go (the one being written too), it is counted and logged once.
    char thrown[160] = "";
    std::string current;   // the file being written: removed with the others when the write fails
    bool stored = false;
    uint64_t b = 0;
    size_t nfiles = 0, on_disk = 0;
    uint64_t disk_bytes = 0;
    try {
        {
            std::lock_guard<std::mutex> lock(m_);
            while (!entries_.empty() && bytes_ + e.bytes > options_.budget_bytes) {
                remove_locked(0, true);
                ++made_room;
            }
            stats_.evicted += made_room;
        }
        t0 = std::chrono::steady_clock::now();
        e.bytes = 0;
        for (size_t k = 0; ok && k < images; ++k) {
            char name[96];
            std::snprintf(name, sizeof name, "%s%016llx-%llu-%zu.bin", file_prefix(), (unsigned long long) nonce_,
                          (unsigned long long) e.id, k);
            current = (fs::path(options_.dir) / name).string();
            size_t written = 0;
            ok = write_file(k, current, written, error);
            if (!ok) break;
            e.files.push_back(current);
            current.clear();
            e.bytes += written;
        }
        if (ok && options_.persist) ok = write_index(e, error);   // after the images: an index names complete files only
        if (ok) {
            b = e.bytes;
            nfiles = e.files.size();
            std::lock_guard<std::mutex> lock(m_);
            // xrig fix #3: the list first (a throw leaves `e` whole for the clean-up below: deque::push_back's strong
            // guarantee), then the counters, so a failed push never counts its bytes
            entries_.push_back(std::move(e));
            stored = true;
            bytes_ += b;
            ++stats_.written;
            stats_.bytes_written += b;
            stats_.write_ms += ms_since(t0);
            on_disk = entries_.size();
            disk_bytes = bytes_;
        }
    } catch (const std::exception& x) {
        describe(thrown, &x);
    } catch (...) {
        describe(thrown, nullptr);
    }
    const double ms = ms_since(t0);
    if (!stored) {
        // xrig fix #3: nothing here throws (remove_file never does), so the clean-up always completes
        if (!current.empty()) remove_file(current);
        if (!e.index.empty()) remove_file(e.index);
        for (const auto& f : e.files) remove_file(f);
        {
            std::lock_guard<std::mutex> lock(m_);
            ++stats_.write_failed;
        }
        try {
            log(fmt("dropped %zu tokens: the write failed after %.0f ms (%s%s)", tokens, ms, thrown[0] ? "threw: " : "",
                    thrown[0] ? thrown : error.c_str()));
        } catch (...) {
        }
        return false;
    }
    try {   // xrig fix #3: stored: a failure to log is not a failed write
        log(fmt("wrote %zu tokens%s: %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s); on disk %zu, %.0f MiB; %zu removed "
                "for the budget", tokens, how, mib(b), nfiles, nfiles == 1 ? "" : "s", ms, mbps(b, ms), on_disk,
                mib(disk_bytes), made_room));
    } catch (...) {
    }
    return true;
}

bool ConversationDiskTier::take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
                                const std::function<bool(uint64_t, std::string&)>& admit,
                                const std::vector<std::vector<SessionKvTarget>>* kv, bool* kept) {
    out = SavedConversation{};
    if (kept) *kept = false;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(m_);
    if (!open_) { error = "the disk tier is off"; return false; }
    // still queued: back from RAM, no disk at all
    for (size_t i = 0; i < queue_.size(); ++i)
        if (queue_[i].entry.id == id) {
            if (kv) {
                error = "still queued: its image is in RAM";
                if (kept) *kept = true;   // it stays in the queue
                return false;
            }
            out = std::move(queue_[i].image);
            const uint64_t b = queue_[i].entry.bytes;
            queue_.erase(queue_.begin() + (std::ptrdiff_t) i);
            ++stats_.ram_hits;
            stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
            lock.unlock();
            log(fmt("RAM hit: %lld tokens back from the write queue (%.0f MiB, no disk) in %.1f ms",
                    (long long) tokens, mib(b), ms_since(t0)));
            log(summary());
            return true;
        }
    // being written: its file is complete when the write ends
    double waited = 0;
    if (writing_ && writing_entry_.id == id) {
        const auto w0 = std::chrono::steady_clock::now();
        cv_.wait(lock, [&] { return !(writing_ && writing_entry_.id == id); });
        waited = ms_since(w0);
    }
    size_t index = entries_.size();
    for (size_t i = 0; i < entries_.size(); ++i)
        if (entries_[i].id == id) index = i;
    if (index == entries_.size()) {
        ++stats_.read_failed;
        error = "the entry is gone (its write failed, or the budget removed it)";
        return false;
    }
    if (kv && kv->size() != entries_[index].files.size()) {
        error = "one set of K/V targets per file is needed";
        return false;
    }
    // xrig fix #4: the entry leaves the list for the read - so the budget (store() on the writer thread) and clear()
    // never delete its files under it - but its files, and with persist its index, stay until the read says what to
    // do with them.  They are deleted after a read that gave the conversation back, or that found them damaged or
    // unreadable.  When only the RAM was missing (SessionError::memory: the RAM preflight refused, or an allocation
    // failed) nothing is wrong with them: the entry is put back and a later request (or the caller's streamed path)
    // still finds it.  Before, a borderline RAM check deleted the conversation for good.
    Entry want = std::move(entries_[index]);   // a move: no allocation before the entry is out of the list
    bytes_ -= want.bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    const uint64_t clears = clears_;   // clear() during the read (#879): the files go after all
    lock.unlock();
    const std::vector<std::string>& files = want.files;
    const size_t nfiles = files.size();
    const SessionFileIdentity fid = file_identity();
    SessionReadLimits limits;
    limits.admit = admit;
    uint64_t bytes = 0;
    bool ok = true, memory = false;
    for (size_t k = 0; ok && k < files.size(); ++k) {
        SavedConversation part;
        size_t n = 0;
        SessionStatus st;
        ok = kv ? session_file_read(files[k], fid, part, (*kv)[k], n, error, limits, &st)
                : session_file_read(files[k], fid, part, n, error, limits, &st);
        bytes += n;
        if (!ok) {
            memory = st.error == SessionError::memory;
            break;
        }
        if (k == 0) out = std::move(part);
        else out.stage_images.push_back(std::move(part));
    }
    if (ok && !holds(out, want.live, want.checkpoints, want.cvec)) {
        ok = false;
        memory = false;   // damaged: never kept
        error = "the files hold another conversation than their index";
    }
    bool keep = !ok && memory;
    auto delete_files = [&] {   // the index first: an index never names a deleted file
        if (!want.index.empty()) remove_file(want.index);
        for (const auto& f : files) remove_file(f);
    };
    if (!keep) delete_files();
    const double ms = ms_since(t0);
    lock.lock();
    if (keep && clears_ != clears) {   // #879 emptied the tier while this read ran: it must not come back
        delete_files();
        keep = false;
    }
    size_t over = 0;
    if (keep) {
        try {
            // back as the newest (it was just asked for); bytes_ only once it is in the list
            entries_.push_back(std::move(want));
            bytes_ += entries_.back().bytes;
        } catch (...) {   // no room for the list's next block: dropped as before, never left orphaned
            delete_files();
            keep = false;
        }
        // what the budget let in while it was out goes first, oldest first (never the one just put back)
        while (keep && entries_.size() > 1 && bytes_ > options_.budget_bytes) {
            remove_locked(0, true);
            ++over;
        }
        stats_.evicted += over;
    }
    if (kept) *kept = keep;
    if (!ok) {
        ++stats_.read_failed;
        lock.unlock();
        out = SavedConversation{};
        log(fmt("read of %lld tokens failed after %.0f ms (%s)%s", (long long) tokens, ms, error.c_str(),
                keep ? "; kept on the disk for a later request (only the RAM was missing)" : ""));
        return false;
    }
    ++stats_.disk_hits;
    stats_.bytes_read += bytes;
    stats_.read_ms += ms - waited;
    stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
    lock.unlock();
    log(fmt("disk hit: %lld tokens, %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s)%s%s", (long long) tokens, mib(bytes),
            nfiles, nfiles == 1 ? "" : "s", ms - waited, mbps(bytes, ms - waited),
            kv ? ", streamed into the GPU" : "",
            waited > 0 ? fmt(", after %.0f ms waiting for its write", waited).c_str() : ""));
    log(summary());
    return true;
}

bool ConversationDiskTier::pending() const {
    std::lock_guard<std::mutex> lock(m_);
    return !queue_.empty() || writing_;
}

void ConversationDiskTier::wait_idle() {
    std::unique_lock<std::mutex> lock(m_);
    if (queue_.empty() && !writing_) return;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t depth = queue_.size() + (writing_ ? 1 : 0);
    cv_.wait(lock, [&] { return queue_.empty() && !writing_; });
    const double ms = ms_since(t0);
    ++stats_.waits;
    stats_.wait_ms += ms;
    lock.unlock();
    log(fmt("waited %.0f ms for %zu queued write%s (their RAM was needed)", ms, depth, depth == 1 ? "" : "s"));
}

bool ConversationDiskTier::wait_idle_for(int64_t max_ms, const std::function<void()>& beat) {
    if (max_ms < 0) { wait_idle(); return true; }
    std::unique_lock<std::mutex> lock(m_);
    if (queue_.empty() && !writing_) return true;
    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(max_ms);
    const size_t depth = queue_.size() + (writing_ ? 1 : 0);
    bool idle = false;
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        const auto slice = std::min<std::chrono::steady_clock::duration>(deadline - now, std::chrono::milliseconds(250));
        if (cv_.wait_for(lock, slice, [&] { return queue_.empty() && !writing_; })) { idle = true; break; }
        if (beat) { lock.unlock(); beat(); lock.lock(); }
    }
    const double ms = ms_since(t0);
    ++stats_.waits;
    stats_.wait_ms += ms;
    if (!idle) ++stats_.wait_timeouts;
    lock.unlock();
    if (beat) beat();
    log(idle ? fmt("waited %.0f ms for %zu queued write%s (their RAM was needed)", ms, depth, depth == 1 ? "" : "s")
             : fmt("gave up waiting after %.0f ms for %zu queued write%s (bound %lld ms): the park is skipped", ms, depth,
                   depth == 1 ? "" : "s", (long long) max_ms));
    return idle;
}

// The disk (or the queue) already holds exactly the state `want`, as an entry's live state or as one of its
// checkpoints: the same tokens and images and control-vector state.  Then writing `want` adds nothing, because every
// prompt it could serve (conversation_prefix: its tokens are a prefix of the prompt) that state serves as well.  A
// LONGER entry is not enough: a prompt that leaves it after `want`'s end does not match its live state.
static bool same_state(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return !a.ids.empty() && a.ids == b.ids && a.imgs == b.imgs;
}

const char* ConversationDiskTier::write_gate(const ConversationCheckpoint& live, bool cvec, uint64_t bytes) {
    if (options_.min_write_tokens && live.ids.size() < options_.min_write_tokens) {
        ++stats_.skipped_short;
        return "under the minimum tokens for a write";
    }
    auto has_state = [&](const Entry& e) {
        if (e.cvec != cvec) return false;
        if (same_state(e.live, live)) return true;
        for (const auto& c : e.checkpoints)
            if (same_state(c, live)) return true;
        return false;
    };
    bool covered = writing_ && has_state(writing_entry_);
    for (const auto& e : entries_) covered = covered || has_state(e);
    for (const auto& j : queue_) covered = covered || has_state(j.entry);
    if (covered) {
        ++stats_.skipped_covered;
        return "the disk already holds it";
    }
    if (options_.day_write_bytes) {
        const auto now = std::chrono::steady_clock::now();
        while (!day_writes_.empty() && now - day_writes_.front().first > std::chrono::hours(24)) {
            day_bytes_ -= day_writes_.front().second;
            day_writes_.pop_front();
        }
        if (day_bytes_ + bytes > options_.day_write_bytes) {
            ++stats_.skipped_day;
            return "over the daily write budget";
        }
        day_writes_.emplace_back(now, bytes);
        day_bytes_ += bytes;
    }
    return nullptr;
}

size_t ConversationDiskTier::size() const {
    std::lock_guard<std::mutex> lock(m_);
    return entries_.size();
}

uint64_t ConversationDiskTier::bytes() const {
    std::lock_guard<std::mutex> lock(m_);
    return bytes_;
}

ConversationDiskStats ConversationDiskTier::stats() const {
    std::lock_guard<std::mutex> lock(m_);
    return stats_;
}

size_t ConversationDiskTier::clear() {
    std::unique_lock<std::mutex> lock(m_);
    const size_t queued = queue_.size();
    queue_.clear();
    cv_.wait(lock, [&] { return !writing_; });
    ++clears_;   // xrig fix #4: a take() reading right now does not put its entry back
    const size_t n = queued + entries_.size();
    while (!entries_.empty()) remove_locked(0, true);
    lock.unlock();
    if (n) log(fmt("removed all %zu conversations (#879)", n));
    return n;
}

std::string ConversationDiskTier::summary() const {
    std::lock_guard<std::mutex> lock(m_);
    const auto& s = stats_;
    return fmt("totals: %zu queued, %zu written (%.0f MiB, %.0f MB/s), %zu RAM hits, %zu disk hits (%.0f MiB, "
               "%.0f MB/s), %llu prompt tokens given back, %zu waits for RAM (%.0f ms), "
               "%zu removed for the budget, %zu/%zu writes/reads failed; on disk %zu (%.0f MiB), queue %zu",
               s.queued, s.written, mib(s.bytes_written), mbps(s.bytes_written, s.write_ms), s.ram_hits, s.disk_hits,
               mib(s.bytes_read), mbps(s.bytes_read, s.read_ms), (unsigned long long) s.tokens_back, s.waits,
               s.wait_ms, s.evicted, s.write_failed, s.read_failed, entries_.size(), mib(bytes_),
               queue_.size() + (writing_ ? 1 : 0));
}

}  // namespace strata::core
