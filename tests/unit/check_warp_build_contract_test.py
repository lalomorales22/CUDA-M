#!/usr/bin/env python3
"""Host-only Warp wrapper checks; mocked tools never clone, build, or download."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_warp_build_contract_test.py <build_warp_cumetal.sh>")
    source = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="cumetal-warp-build-contract-") as temporary:
        root = Path(temporary).resolve()
        scripts = root / "scripts"
        (scripts / "warp-patches").mkdir(parents=True)
        script = scripts / source.name
        shutil.copyfile(source, script)
        tools = root / "tools"
        tools.mkdir()
        checkout = root / "Warp checkout"
        (checkout / ".git").mkdir(parents=True)
        build = root / "CuMetal build"
        (build / "cumetal-cuda-toolkit/bin").mkdir(parents=True)
        (build / "libcumetal.dylib").touch()
        write_executable(build / "cumetal-cuda-toolkit/bin/nvcc", "#!/bin/sh\nexit 91\n")
        log = root / "commands.jsonl"
        recorder = """#!/usr/bin/env python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open(os.environ['WARP_TEST_COMMANDS'], 'a') as output:
    output.write(json.dumps({'tool': name, 'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')
if name == 'git':
    if 'clone' in sys.argv:
        raise SystemExit('unexpected clone: this test never permits downloads')
    if sys.argv[1:] != ['-C', os.environ['CUMETAL_WARP_DIR'], 'rev-parse', '--short', 'HEAD']:
        raise SystemExit('unexpected git command')
    print('e6c3ba2')
"""
        write_executable(tools / "git", recorder)
        python = tools / "existing python"
        write_executable(python, recorder)
        write_executable(scripts / "warp-patches/apply_warp_patches.sh",
                         "#!/bin/sh\nexec \"${WARP_TEST_PATCH_RECORDER}\" \"$@\"\n")
        patch_recorder = tools / "patch-recorder"
        write_executable(patch_recorder, recorder)
        base = {key: value for key, value in os.environ.items() if not key.startswith("CUMETAL_")}
        base.update(PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
                    CUMETAL_WARP_DIR=str(checkout), CUMETAL_BUILD_DIR=str(build),
                    CUMETAL_PYTHON=str(python), CUMETAL_JOBS="2",
                    WARP_TEST_COMMANDS=str(log), WARP_TEST_PATCH_RECORDER=str(patch_recorder))
        cases = []

        def run(name, settings=None, arguments=("--build",), failure=None):
            log.unlink(missing_ok=True)
            env = dict(base)
            env.update(settings or {})
            result = subprocess.run(["bash", str(script), *arguments], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=20)
            commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            if failure is not None:
                assert result.returncode == 2 and failure in result.stderr, (name, result)
                assert not commands, (name, "validation occurred after fetch/patch/build", commands)
            else:
                assert result.returncode == 0, (name, result)
            cases.append(name)
            return result, commands

        def check_build(commands, standalone, llvm=None):
            assert [record["tool"] for record in commands] == ["git", "patch-recorder", "existing python"]
            expected = ["build_lib.py", f"--cuda-path={build}/cumetal-cuda-toolkit", "--no-use-libmathdx"]
            expected += ["--standalone", f"--llvm-path={llvm}"] if standalone else ["--no-standalone"]
            expected += ["--jobs=2"]
            assert commands[-1]["args"] == expected, commands
            assert commands[-1]["cwd"] == str(checkout), commands

        result, commands = run("default-no-standalone")
        check_build(commands, False)
        assert "CPU JIT is not built" in result.stdout
        _, commands = run("explicit-no-standalone", {"CUMETAL_WARP_STANDALONE": "0"})
        check_build(commands, False)
        for value in ("", "2", "true", "0 --standalone", "$(touch sentinel)"):
            run("invalid-standalone-" + repr(value), {"CUMETAL_WARP_STANDALONE": value},
                failure="CUMETAL_WARP_STANDALONE must be 0 or 1")
        run("missing-llvm", {"CUMETAL_WARP_STANDALONE": "1"}, failure="existing CUMETAL_WARP_LLVM_PATH")
        run("absent-llvm", {"CUMETAL_WARP_STANDALONE": "1", "CUMETAL_WARP_LLVM_PATH": str(root / "absent")},
            failure="existing CUMETAL_WARP_LLVM_PATH")
        llvm = root / "llvm-existing"
        llvm.mkdir()
        run("empty-llvm-prefix", {"CUMETAL_WARP_STANDALONE": "1", "CUMETAL_WARP_LLVM_PATH": str(llvm)},
            failure="existing CUMETAL_WARP_LLVM_PATH")
        (llvm / "include").mkdir()
        run("llvm-without-libraries", {"CUMETAL_WARP_STANDALONE": "1", "CUMETAL_WARP_LLVM_PATH": str(llvm)},
            failure="existing CUMETAL_WARP_LLVM_PATH")
        (llvm / "lib").mkdir()
        _, commands = run("existing-llvm-prefix", {"CUMETAL_WARP_STANDALONE": "1",
                                                  "CUMETAL_WARP_LLVM_PATH": str(llvm)})
        check_build(commands, True, llvm)
        _, commands = run("relative-llvm-prefix", {"CUMETAL_WARP_STANDALONE": "1",
                                                  "CUMETAL_WARP_LLVM_PATH": llvm.name})
        check_build(commands, True, llvm)
        for suffix in (" with-spaces", "$(touch sentinel)", "`touch sentinel`", ";touch-sentinel",
                       "&sentinel", "|sentinel", '"quoted"', "'quoted'", "\\escape", "\ttab", "\nnewline", "\n"):
            unsafe = root / ("llvm" + suffix)
            (unsafe / "include").mkdir(parents=True)
            (unsafe / "lib").mkdir()
            run("unsafe-llvm-" + repr(suffix), {"CUMETAL_WARP_STANDALONE": "1",
                                               "CUMETAL_WARP_LLVM_PATH": str(unsafe)},
                failure="pinned Warp cannot safely handle spaces or shell metacharacters")
        llvm_link = root / "llvm-link"
        llvm_link.symlink_to(root / "llvm\n", target_is_directory=True)
        run("unsafe-canonical-llvm-path", {"CUMETAL_WARP_STANDALONE": "1",
                                            "CUMETAL_WARP_LLVM_PATH": str(llvm_link)},
            failure="pinned Warp cannot safely handle spaces or shell metacharacters")
        run("llvm-with-standalone-disabled", {"CUMETAL_WARP_LLVM_PATH": str(llvm)},
            failure="requires CUMETAL_WARP_STANDALONE=1")
        run("invalid-before-clone", {"CUMETAL_WARP_DIR": str(root / "uncloned"),
                                     "CUMETAL_WARP_STANDALONE": "1"}, arguments=("--clone-only",),
            failure="existing CUMETAL_WARP_LLVM_PATH")
        _, commands = run("clone-only-no-build", arguments=("--clone-only",))
        assert [record["tool"] for record in commands] == ["git", "patch-recorder"]
        run("unexpected-mode", arguments=("--unknown",), failure="usage:")
        run("trailing-cli-argument", arguments=("--build", "--standalone"), failure="usage:")
        assert not (root / "sentinel").exists(), "environment values were evaluated as shell syntax"
    print(f"PASS: {len(cases)} Warp build contract cases; no downloads, builds, or GPU execution")


if __name__ == "__main__":
    main()
