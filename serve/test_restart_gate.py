"""xrig incidents 1 and 3 (node.3.2, 2-Nov): a start waits for the previous engine's VRAM and RAM, and the engine runs
with a core-file limit of 1."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server
from serve.server import ChatTemplate, EngineReleasing, GpuBusy, Service, StrataEngine, serve
from serve.test_server import ByteTokenizer

try:
    import resource
except ImportError:
    resource = None

FAKE = r'''
import sys
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    if line.startswith(("GEN", "BGEN")):
        print("T 90\nDONE 1 1 0 1 stop", flush=True)
'''


def core_limits(pid):
    """(soft, hard) of "Max core file size" in /proc/<pid>/limits, as text."""
    with open(f"/proc/{pid}/limits", encoding="ascii") as f:
        for line in f:
            if line.startswith("Max core file size"):
                return tuple(line.split()[4:6])
    raise AssertionError("no core limit line")


def as_text(limit):
    return "unlimited" if limit == resource.RLIM_INFINITY else str(limit)


def readings(*values):
    """A reader that returns `values` in turn, then the last one again."""
    it = iter(values)
    last = [None]

    def read(*_):
        last[0] = next(it, last[0])
        return last[0]
    return read


class Gate(unittest.TestCase):
    def setUp(self):
        self.notes = []
        for name, value in (("RESTART_POLL_S", 0.01), ("RESTART_SETTLE_S", 0.01), ("RESTART_WAIT_S", 5.0),
                            ("RESTART_KILL_GRACE_S", 0.05)):
            p = mock.patch.object(server, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(StrataEngine, "_gate_note", lambda _s, _log, text: self.notes.append(text))
        p.start()
        self.addCleanup(p.stop)

    def engine(self, gate=None):
        e = StrataEngine("missing-executable", [], lazy=True)
        if gate is not None:
            e._gate = dict(gate)
        return e

    def run_gate(self, e, vram, avail):
        with mock.patch.object(server, "vram_used_mib", vram), mock.patch.object(server, "mem_available_mib", avail):
            e._wait_for_memory(None, None)

    def test_first_start_takes_the_baseline_once_memory_settles(self):
        e = self.engine()
        self.run_gate(e, readings(9000, 5000, 1000, 1000), readings(20000, 30000, 40000, 40000))
        self.assertEqual(e._gate["vram0"], 1000)
        self.assertEqual(e._gate["avail_pre"], 40000)
        self.assertIsNone(e._gate["need"])
        self.assertEqual(sum("still coming back" in n for n in self.notes), 2, self.notes)
        self.assertIn("VRAM used 1000 MiB, MemAvailable 40000 MiB", self.notes[-1])
        self.assertTrue(self.notes[-1].startswith("start: baseline before the first start after"), self.notes)

    def test_first_start_waits_while_another_process_has_dev_kfd_open(self):
        e = self.engine()
        old = [(4321, "strata", 1)]
        with mock.patch.object(server, "other_kfd_users", readings(old, old, [])):
            self.run_gate(e, readings(1000), readings(40000))     # memory steady: only the old engine holds it
        waits = [n for n in self.notes if "pid 4321 (strata) has /dev/kfd open" in n]
        self.assertEqual(len(waits), 2, self.notes)
        self.assertIn("other /dev/kfd users: none", self.notes[-1])
        self.assertEqual(e._gate["vram0"], 1000)

    def test_a_process_that_keeps_dev_kfd_open_holds_the_first_start_only_until_the_wait_ends(self):
        e = self.engine()
        with mock.patch.object(server, "RESTART_WAIT_S", 0.05), \
                mock.patch.object(server, "other_kfd_users", lambda: [(777, "ollama", 1)]):
            self.run_gate(e, readings(1000), readings(40000))
        self.assertTrue(any("has /dev/kfd open" in n for n in self.notes), self.notes)
        self.assertIn("other /dev/kfd users: pid 777 (ollama)", self.notes[-1])
        self.assertEqual(e._gate["vram0"], 1000)

    def test_first_start_without_a_readable_gpu_does_not_wait(self):
        e = self.engine()
        t0 = time.monotonic()
        with mock.patch.object(server, "RESTART_SETTLE_S", 5.0):
            self.run_gate(e, readings(None), readings(40000))
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertIsNone(e._gate["vram0"])

    def test_waits_until_vram_and_ram_are_back_and_logs_every_reading(self):
        e = self.engine({"vram0": 1000, "need": 30000, "prev": None, "avail_pre": None})
        self.run_gate(e, readings(20000, 8000, 1200), readings(9000, 20000, 40000))
        waits = [n for n in self.notes if n.startswith("start: waiting")]
        self.assertEqual(len(waits), 2, self.notes)
        self.assertIn("VRAM used 20000 MiB (at most 1512 MiB), MemAvailable 9000 MiB (needs 30000 MiB)", waits[0])
        self.assertIn("VRAM used 8000 MiB (at most 1512 MiB), MemAvailable 20000 MiB (needs 30000 MiB)", waits[1])
        self.assertIn("memory back after", self.notes[-1])
        self.assertIn("VRAM used 1200 MiB", self.notes[-1])
        self.assertEqual(e._gate["avail_pre"], 40000)

    def test_ram_alone_holds_the_start(self):
        e = self.engine({"vram0": 1000, "need": 30000, "prev": None, "avail_pre": None})
        self.run_gate(e, readings(1000), readings(9000, 31000))
        self.assertEqual(sum(n.startswith("start: waiting") for n in self.notes), 1, self.notes)

    def test_ram_flat_below_the_need_starts_once_settled(self):
        # other programs' use keeps MemAvailable below what the last load took: no old engine, VRAM back, RAM flat
        e = self.engine({"vram0": 1000, "need": 30000, "prev": None, "avail_pre": None})
        self.run_gate(e, readings(1000), readings(9000))
        self.assertEqual(sum(n.startswith("start: waiting") for n in self.notes), 2, self.notes)
        self.assertTrue(self.notes[-1].startswith("start: RAM settled at 9000 MiB, below the last load's need "
                                                  "30000 MiB"), self.notes)
        self.assertIn("the engine's memory guard decides", self.notes[-1])

    def test_ram_still_rising_keeps_the_start_waiting(self):
        e = self.engine({"vram0": 1000, "need": 30000, "prev": None, "avail_pre": None})
        self.run_gate(e, readings(1000), readings(9000, 12000, 15000, 18000, 18100, 18150))
        self.assertEqual(sum(n.startswith("start: waiting") for n in self.notes), 5, self.notes)
        self.assertTrue(self.notes[-1].startswith("start: RAM settled at 18150 MiB"), self.notes)

    def test_the_first_starts_wait_logs_ten_steps_then_how_it_ended(self):
        e = self.engine()
        t0 = time.monotonic()
        with mock.patch.object(server, "RESTART_KFD_WAIT_S", 0.3), \
                mock.patch.object(server, "other_kfd_users", lambda: [(777, "ollama", 1)]):
            self.run_gate(e, readings(1000), readings(40000))
        self.assertLess(time.monotonic() - t0, 2.0)       # a program that stays: at most RESTART_KFD_WAIT_S
        waits = [n for n in self.notes if "has /dev/kfd open" in n]
        self.assertEqual(len(waits), server.RESTART_FIRST_LOG_LINES, self.notes)
        self.assertIn("further steps are not logged", waits[-1])
        self.assertTrue(self.notes[-1].startswith("start: baseline before the first start after"), self.notes)
        self.assertIn("other /dev/kfd users: pid 777 (ollama)", self.notes[-1])

    def test_memory_back_at_once_is_one_line(self):
        e = self.engine({"vram0": 1000, "need": 30000, "prev": None, "avail_pre": None})
        self.run_gate(e, readings(1100), readings(50000))
        self.assertEqual(len(self.notes), 1)
        self.assertTrue(self.notes[0].startswith("start: memory back: VRAM used 1100 MiB (at most 1512 MiB)"),
                        self.notes)

    def test_unreadable_readings_are_said_so(self):
        self.assertEqual(server.memory_words(None, None, 1000, 5, bounds=True),
                         "VRAM not readable, MemAvailable not readable")
        self.assertEqual(server.memory_words(800, 9000, None, None, bounds=True),
                         "VRAM used 800 MiB (no baseline), MemAvailable 9000 MiB (no load measured yet)")

    def test_past_the_wait_the_start_is_refused_with_a_503(self):
        e = self.engine({"vram0": 1000, "need": None, "prev": None, "avail_pre": None})
        with mock.patch.object(server, "RESTART_WAIT_S", 0.05):
            with self.assertRaises(EngineReleasing) as cm:
                self.run_gate(e, readings(20000), readings(50000))
        self.assertIsInstance(cm.exception, GpuBusy)    # every GpuBusy handler answers 503 + Retry-After
        self.assertEqual(cm.exception.retry_after, 15)
        self.assertIn("VRAM used 20000 MiB (at most 1512 MiB)", str(cm.exception))
        self.assertTrue(self.notes[-1].startswith("start refused"), self.notes)

    def test_off_never_waits(self):
        e = self.engine({"vram0": 1000, "need": 30000, "prev": None, "avail_pre": None})
        with mock.patch.object(server, "RESTART_WAIT_S", 0.0):
            self.run_gate(e, readings(20000), readings(100))
        self.assertEqual(len(self.notes), 1)

    @unittest.skipUnless(os.path.exists("/proc/self/stat"), "needs /proc")
    def test_a_previous_engine_still_running_is_killed_after_the_grace(self):
        old = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: old.poll() is None and old.kill())
        e = self.engine({"vram0": None, "need": None, "prev": (old.pid, server.process_start(old.pid)),
                         "avail_pre": None})
        self.run_gate(e, readings(None), readings(None))
        self.assertEqual(old.wait(timeout=5), -9)
        self.assertTrue(any("was still running" in n and "killed" in n for n in self.notes), self.notes)
        self.assertIn("memory back after", self.notes[-1])

    @unittest.skipUnless(os.path.exists("/proc/self/stat"), "needs /proc")
    def test_process_alive_checks_the_start_time(self):
        me = os.getpid()
        self.assertTrue(server.process_alive(me, server.process_start(me)))
        self.assertFalse(server.process_alive(me, "1"))            # the same pid, another process
        self.assertFalse(server.process_alive(2 ** 22 + 12345, None))


class VramReading(unittest.TestCase):
    def test_the_cards_the_engine_sees(self):
        got = {}

        def card(i, amd):
            got.setdefault(amd, []).append(i)
            return (i + 1) << 30
        with mock.patch.object(server, "_card_vram_used", card):
            self.assertEqual(server.vram_used_mib({"HIP_VISIBLE_DEVICES": "0,2"}), 4096)
            self.assertEqual(got, {True: [0, 2]})
            self.assertEqual(server.vram_used_mib({"CUDA_VISIBLE_DEVICES": "1"}), 2048)
            self.assertIsNone(server.vram_used_mib({"CUDA_VISIBLE_DEVICES": "GPU-5b1e"}))

    def test_every_card_found_without_a_visible_list(self):
        with mock.patch.object(server, "_card_vram_used", lambda i, amd: (1 << 30) if amd and i < 2 else None):
            self.assertEqual(server.vram_used_mib({}), 2048)
        with mock.patch.object(server, "_card_vram_used", lambda i, amd: None):
            self.assertIsNone(server.vram_used_mib({}))


class RealStart(unittest.TestCase):
    """A start through the real pipe protocol with a fake engine."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        py = Path(tmp.name) / "native.py"
        if os.name == "nt":
            py.write_text(FAKE)
            self.script = Path(tmp.name) / "native.cmd"
            self.script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            py.write_text("#!" + sys.executable + "\n" + FAKE)
            py.chmod(0o700)
            self.script = py
        p = mock.patch.object(server, "vram_used_mib", lambda env: None)
        p.start()
        self.addCleanup(p.stop)

    def start(self):
        e = StrataEngine(str(self.script), [])
        self.addCleanup(e.close)
        return e

    def test_need_is_what_the_last_load_took(self):
        printed = []
        with mock.patch.object(server, "mem_available_mib", readings(50000, 37000)), \
                mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            e = self.start()
        self.assertEqual(e._gate["need"], 13000)
        self.assertEqual(e._gate["prev"][0], e.proc.pid)
        self.assertTrue(any("start: this load took 13000 MiB of MemAvailable (50000 -> 37000 MiB); the next start "
                            "waits for that much, or for RAM to settle" in p for p in printed), printed)

    def test_need_is_not_capped(self):
        with mock.patch.object(server, "mem_available_mib", readings(50000, 1000)):
            e = self.start()
        self.assertEqual(e._gate["need"], 49000)

    @unittest.skipUnless(resource is not None and hasattr(resource, "prlimit"), "needs prlimit (Linux)")
    def test_a_core_limit_that_cannot_be_set_is_logged_and_the_start_goes_on(self):
        printed = []
        with mock.patch.object(resource, "prlimit", mock.Mock(side_effect=PermissionError("not allowed"))), \
                mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))):
            e = self.start()
        self.assertTrue(e.alive())
        self.assertTrue(any("core-file limit" in p and "could not be set" in p for p in printed), printed)

    def test_a_refused_start_is_a_503_and_the_next_request_starts_it(self):
        e = self.start()
        svc = Service(e, ByteTokenizer(), ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        http = serve(svc, port=0)
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        base = f"http://127.0.0.1:{http.server_address[1]}"
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()

        def post():
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        e.proc.kill()                                     # it died; the next request starts it again
        end = time.monotonic() + 5
        while e.alive() and time.monotonic() < end:
            time.sleep(0.01)
        e._gate["vram0"] = 1000
        vram = [20000]                                    # the old engine's VRAM is not back
        with mock.patch.object(server, "RESTART_WAIT_S", 0.2), mock.patch.object(server, "RESTART_POLL_S", 0.05), \
                mock.patch.object(server, "vram_used_mib", lambda env: vram[0]):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                post()
            self.assertEqual(cm.exception.code, 503)
            self.assertEqual(cm.exception.headers.get("Retry-After"), "15")
            self.assertIn("memory is not back", cm.exception.read().decode())
            vram[0] = 1000                                # the memory is back
            status, answer = post()
        self.assertEqual(status, 200)
        self.assertEqual(answer["usage"]["completion_tokens"], 1)
        self.addCleanup(svc.engine.close)

    def test_restart_does_not_spend_a_try_on_the_wait(self):
        e = self.start()
        calls = []

        def refuse(*_):
            calls.append(1)
            raise EngineReleasing("not back", retry_after=15)
        with mock.patch.object(StrataEngine, "_wait_for_memory", refuse):
            with self.assertRaises(EngineReleasing):
                e.restart()
        self.assertEqual(len(calls), 1)                  # no second try, no RESTART_RETRY_S sleep
        e.restart()                                      # the next request waits again, and starts it
        self.assertTrue(e.alive())

    @unittest.skipUnless(resource is not None and hasattr(resource, "prlimit"), "needs prlimit (Linux)")
    def test_the_engine_runs_with_a_core_limit_of_one_and_the_server_keeps_its_own(self):
        before = resource.getrlimit(resource.RLIMIT_CORE)
        if before[1] == 0:
            self.skipTest("a hard core limit of 0: no core at all, nothing to set")
        e = self.start()
        self.assertEqual(core_limits(e.proc.pid), ("1", as_text(before[1])))
        self.assertEqual(resource.getrlimit(resource.RLIMIT_CORE), before)

    @unittest.skipUnless(resource is not None and hasattr(resource, "prlimit"), "needs prlimit (Linux)")
    def test_keep_leaves_the_engine_the_servers_limit(self):
        before = resource.getrlimit(resource.RLIMIT_CORE)
        with mock.patch.object(server, "ENGINE_CORE_KEEP", True):
            e = self.start()
        self.assertEqual(core_limits(e.proc.pid), (as_text(before[0]), as_text(before[1])))

    @unittest.skipUnless(resource is not None and hasattr(resource, "prlimit"), "needs prlimit (Linux)")
    def test_no_core_dumps_sets_any_started_process_and_survives_one_that_ended(self):
        if resource.getrlimit(resource.RLIMIT_CORE)[1] == 0:
            self.skipTest("a hard core limit of 0")
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: p.poll() is None and p.kill())
        server.no_core_dumps(p)                           # the image encoder takes this same call
        self.assertEqual(core_limits(p.pid)[0], "1")
        p.kill()
        p.wait(timeout=5)
        server.no_core_dumps(p)                           # gone: no error
        server.no_core_dumps(object())                    # a stand-in without a pid (the tests' fake processes)


if __name__ == "__main__":
    unittest.main()
