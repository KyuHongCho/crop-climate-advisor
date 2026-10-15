"""Importing any `evals` module makes no subprocess, network or model call.

Runs in a fresh, isolated interpreter so modules already imported by the suite
cannot hide an import-time side effect.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
import json, os, pkgutil, socket, subprocess, urllib.request

calls = []

def _forbidden(name):
    def deny(*args, **kwargs):
        calls.append(name)
        raise RuntimeError(f"import-time side effect: {name}")
    return deny

# Subclasses rather than functions: ssl and others subclass these at import time.
class _NoSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        _forbidden("socket.socket")()

class _NoPopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        _forbidden("subprocess.Popen")()

socket.socket = _NoSocket
socket.getaddrinfo = _forbidden("socket.getaddrinfo")
subprocess.Popen = _NoPopen
subprocess.run = _forbidden("subprocess.run")
os.system = _forbidden("os.system")
urllib.request.urlopen = _forbidden("urllib.request.urlopen")

import evals
names = ["evals"] + [m.name for m in pkgutil.walk_packages(evals.__path__, "evals.")]
for name in names:
    __import__(name)
print(json.dumps({"modules": names, "calls": calls}))
"""


def _import_all(root):
    proc = subprocess.run([sys.executable, "-I", "-c", _CHILD, str(root)],
                          capture_output=True, text=True, timeout=60)
    report = json.loads(proc.stdout.splitlines()[-1]) if proc.returncode == 0 else None
    return proc, report


class EvalImportsTest(unittest.TestCase):
    def test_every_evals_module_imports_without_side_effects(self):
        proc, report = _import_all(ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["calls"], [])
        self.assertIn("evals.record_fixtures", report["modules"])

    def test_a_swallowed_forbidden_call_is_still_reported(self):
        attempts = {
            "urllib.request.urlopen": "import urllib.request; urllib.request.urlopen('http://1.1.1.1/')",
            "socket.getaddrinfo": "import socket; socket.getaddrinfo('power.larc.nasa.gov', 443)",
            "os.system": "import os; os.system('true')",
            "subprocess.Popen": "import subprocess; subprocess.Popen(['true'])",
            "subprocess.run": "import subprocess; subprocess.check_output(['true'])",
        }
        for name, attempt in attempts.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                (Path(d) / "evals").mkdir()
                (Path(d) / "evals" / "__init__.py").write_text(
                    f"try:\n    {attempt}\nexcept Exception:\n    pass\n")
                proc, report = _import_all(d)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(report["calls"], [name])


if __name__ == "__main__":
    unittest.main()
