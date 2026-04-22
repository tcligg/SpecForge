"""Docker image build/push helper.

Extracted verbatim from gke/deploy.py in step 2. No logic changes.

Rename: _build_and_push_image -> build_and_push_image.
gke/deploy.py re-exports the old underscore name for compat.
"""

from __future__ import annotations

import subprocess
import sys

from gke.lib.config import SCRIPT_DIR, Config


def build_and_push_image(cfg: Config) -> str:
    """Builds and pushes the Docker image, returning the new image URI."""
    print()
    print("Building and pushing new image...")
    build_script = str(SCRIPT_DIR / "build_and_push.sh")
    project_root = SCRIPT_DIR.parent
    result = subprocess.run(
        ["bash", build_script],
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root,
    )
    print(result.stdout)

    new_image_uri = None
    for line in result.stdout.splitlines():
        if "Full tag:" in line:
            new_image_uri = line.split("Full tag:")[-1].strip()
            break

    if not new_image_uri:
        print("Error: Could not extract 'Full tag:' from build output.")
        sys.exit(1)

    return new_image_uri
