#!/bin/sh
# Build the Windows release of a commit, from WSL on the Windows PC:
#   tools/build.sh [REF]          (default HEAD)
# The commit is exported to %USERPROFILE%\Projects\ito\snap\NAME and built into
# %USERPROFILE%\Projects\ito\builds\NAME\ito (NAME: the branch, slashes as dashes).
# Windows needs uv and Visual Studio with C++.
set -eu
ref=${1:-HEAD}
commit=$(git rev-parse --short "$ref^{commit}")
name=$(git rev-parse --abbrev-ref "$ref" | sed 's#^origin/##; s#/#-#g')
[ "$name" = HEAD ] && name=$commit
win=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")/Projects/ito
snap=$win/snap/$name
rm -rf "$snap" "$win/builds/$name"
mkdir -p "$snap" "$win/builds"
git archive "$commit" | tar -x -C "$snap"
echo "building $commit as $name"
cd "$snap"
UV_CACHE_DIR=$(wslpath -w "$win/uv-cache") UV_PROJECT_ENVIRONMENT=$(wslpath -w "$win/venv") \
WSLENV=UV_CACHE_DIR:UV_PROJECT_ENVIRONMENT \
    cmd.exe /c uv run -q --python 3.12 --no-dev python release\\windows.py \
    --dist "$(wslpath -w "$win/builds/$name")"
