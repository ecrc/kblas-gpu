#!/usr/bin/env python3
# Checks that the CMake build lists the same library sources as the Makefile build,
# and that every public header is installed. Prints each mismatch and exits with 1 if there is any.
# Usage: check_cmake_sources.py [repository root]
import os
import re
import subprocess
import sys

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
src = os.path.join(root, "src")

# Sources the Makefile build compiles with every optional part enabled; a dry run needs no toolkit
flags = ["_SUPPORT_BLAS2_=TRUE", "_SUPPORT_BLAS3_=TRUE", "_SUPPORT_BATCH_TR_=TRUE", "_SUPPORT_TLR_=TRUE",
         "_SUPPORT_SVD_=TRUE", "_SUPPORT_LAPACK_=TRUE", "_USE_MAGMA_=TRUE",
         "_CUDA_ROOT_=/unused", "_MAGMA_ROOT_=/unused", "NVCC=nvcc"]
env = {k: v for k, v in os.environ.items() if k not in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL")}
run = subprocess.run(["make", "-n", "-B", "-C", src] + flags, env=env, capture_output=True, text=True)
if run.returncode != 0:
    sys.exit("make dry run failed:\n" + run.stderr)
make_srcs = {os.path.normpath(f) for f in re.findall(r"\s-c\s+(\S+)", run.stdout)}

# Sources named in the CMake lists under src
cmake_srcs = set()
for d, _, files in os.walk(src):
    if "CMakeLists.txt" in files:
        text = re.sub(r"#.*", "", open(os.path.join(d, "CMakeLists.txt")).read())
        for f in re.findall(r"[\w./-]+\.(?:cu|cpp|c)\b", text):
            cmake_srcs.add(os.path.relpath(os.path.join(d, f), src))

# Public headers and the headers the CMake build installs
inc = os.path.join(root, "include")
headers = {f for f in os.listdir(inc) if f.endswith(".h")}
installed = set(re.findall(r"[\w./-]+\.h\b", re.sub(r"#.*", "", open(os.path.join(inc, "CMakeLists.txt")).read())))

problems = []
problems += ["built by the Makefiles but missing from CMake: src/" + f for f in sorted(make_srcs - cmake_srcs)]
problems += ["listed in CMake but not built by the Makefiles: src/" + f for f in sorted(cmake_srcs - make_srcs)]
problems += ["header not installed by CMake: include/" + f for f in sorted(headers - installed)]
problems += ["CMake installs a header that does not exist: include/" + f for f in sorted(installed - headers)]
for p in problems:
    print(p)
if problems:
    sys.exit(1)
print("CMake lists match the Makefiles: %d sources, %d headers" % (len(make_srcs), len(headers)))
