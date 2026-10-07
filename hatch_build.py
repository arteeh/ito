"""Ship the generated bundle, tagged for the Python/platform that built its kernels."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        bundle = Path(self.root) / "ito" / "_slam"
        if version != "editable" and bundle.is_dir() and self.target_name == "wheel":
            if not (bundle / "manifest.json").is_file():
                raise RuntimeError(
                    "Incomplete SLAM build: run python -m ito.build before packaging"
                )
            build_data["force_include"][str(bundle)] = "ito/_slam"
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
