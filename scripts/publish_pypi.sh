#!/usr/bin/env bash
# Publish the current dist/ to PyPI with an API token kept OUTSIDE git.
#
#   1. create a token at https://pypi.org/manage/account/token/  (scope: "entire account" for the
#      first upload; afterwards you can scope it to the any2jev project)
#   2. save it as pypi_token.txt next to this repo's pyproject.toml (the file is gitignored)
#   3. bash scripts/publish_pypi.sh            # builds, checks, uploads, verifies
#
# Alternative without any local token: publish a GitHub release and let .github/workflows/publish.yml
# upload through PyPI trusted publishing.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1   # twine's progress bar uses characters a GBK console cannot print

TOKEN_FILE="${PYPI_TOKEN_FILE:-pypi_token.txt}"
if [[ -z "${PYPI_TOKEN:-}" ]]; then
  [[ -f "$TOKEN_FILE" ]] || { echo "no PYPI_TOKEN in the environment and no $TOKEN_FILE; see the header of this script"; exit 1; }
  PYPI_TOKEN="$(tr -d '\r\n ' < "$TOKEN_FILE")"
fi
[[ "$PYPI_TOKEN" == pypi-* ]] || { echo "the token must start with 'pypi-' (recovery codes and passwords do not work for uploads)"; exit 1; }

VERSION="$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
echo "building any2jev $VERSION"
rm -rf dist build
python -m build --wheel --sdist -o dist/ >/dev/null
python -m twine check dist/*
if tar tzf "dist/any2jev-$VERSION.tar.gz" | grep -q -i -E "key|token|recovery|\.env"; then
  echo "refusing to upload: a secret-looking file is inside the sdist"; exit 1
fi
python -m twine upload --non-interactive -u __token__ -p "$PYPI_TOKEN" dist/*
echo "uploaded; waiting for the index"
for i in $(seq 1 12); do
  if curl -sf "https://pypi.org/pypi/any2jev/$VERSION/json" >/dev/null; then
    echo "https://pypi.org/project/any2jev/$VERSION/ is live"; exit 0
  fi
  sleep 10
done
echo "upload finished but the index has not caught up yet; check https://pypi.org/project/any2jev/"
