"""Stage and pilot offline: uv run --extra slam python e2e/bundle.py."""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="ito-release-") as temporary:
        directory = Path(temporary)
        subprocess.run(["uv", "build", "--out-dir", temporary], cwd=ROOT, check=True)
        (wheel,) = directory.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            assert not any(
                "models/" in name or "_slam/" in name or ".slam-build/" in name
                for name in archive.namelist()
            ), "Models must ship alongside the executable"
        app = directory / "app"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(app)], check=True)
        scripts = app / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            check=True,
        )
        # Share already installed dependencies; Ito itself must come from the wheel.
        site = Path(
            subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        (site / "e2e-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n")
        models = scripts / "models"
        subprocess.run(
            [sys.executable, "-m", "ito.build", "--output", str(models)],
            cwd=ROOT,
            check=True,
        )
        manifest = json.loads((models / "manifest.json").read_text())
        for name, size in manifest["files"].items():
            assert (models / name).stat().st_size == size, name
        for name in ("LICENSE", "NOTICE", "CHECKPOINTS_NOTICE", "ATTRIBUTION.txt"):
            assert (models / name).read_text(), name
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                str(python),
                "-c",
                "from ito.reconstruction.mast3r_runtime import MODELS; "
                "import ito, sys; from pathlib import Path; "
                "assert Path(ito.__file__).is_relative_to(sys.prefix), ito.__file__; "
                "assert MODELS == Path(sys.argv[1]), MODELS",
                str(models),
            ],
            cwd=temporary,
            env=env,
            check=True,
        )
        cases = [["--models", case] for case in ("missing", "corrupt", "present")]
        if manifest["native"]:
            import torch

            if torch.cuda.is_available():
                cases[-1] = ["--cuda"]
        for args in cases:
            subprocess.run(
                [str(python), str(ROOT / "e2e/slam.py"), *args],
                cwd=temporary,
                env=env,
                check=True,
            )
        output = ROOT / "e2e/out/bundle"
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(directory / "e2e/out", output, dirs_exist_ok=True)
    print("PASS: release assets, licences, installed app, missing/corrupt models, offline pilot")


if __name__ == "__main__":
    main()
