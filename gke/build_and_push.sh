#!/usr/bin/env bash
# Build the SpecForge regen Docker image and push to Google Artifact Registry.
#
# Usage:
#   bash gke/build_and_push.sh                  # uses defaults
#   bash gke/build_and_push.sh my-tag           # custom tag
#
# Environment variables (override defaults):
#   PROJECT       - GCP project ID           (default: pyc-vtx-dev)
#   REGION        - AR region                (default: us-central1)
#   REPO          - AR repository name       (default: pyc-vtx-us-central1)
#   IMAGE_NAME    - Image name               (default: specforge-regen)
#   TAG           - Image tag                (default: YYYYMMDD-HHMMSS, or $1)

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(dirname "$SCRIPT_DIR")

PROJECT="${PROJECT:-pyc-vtx-dev}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-pyc-vtx-us-central1}"
IMAGE_NAME="${IMAGE_NAME:-specforge-regen}"
TAG="${1:-${TAG:-$(date +%Y%m%d-%H%M%S)}}"

REGISTRY="${REGION}-docker.pkg.dev"
FULL_IMAGE="${REGISTRY}/${PROJECT}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "============================================================"
echo "  SpecForge Docker Build & Push"
echo "============================================================"
echo "  Registry:  ${REGISTRY}"
echo "  Project:   ${PROJECT}"
echo "  Repo:      ${REPO}"
echo "  Image:     ${IMAGE_NAME}:${TAG}"
echo "  Full tag:  ${FULL_IMAGE}"
echo "  Context:   ${REPO_ROOT}"
echo "============================================================"

# Configure Docker to authenticate with Artifact Registry
echo ""
echo "Configuring Docker auth for ${REGISTRY}..."
gcloud auth configure-docker "${REGISTRY}" --quiet

# Build
#
# --pull forces docker to re-fetch the FROM image (lmsysorg/sglang:dev)
# from the registry on every build, ignoring the local layer cache for
# the base. We need this because :dev is a moving tag — a stale local
# copy means we ship an outdated sglang against the latest transformers,
# which surfaces as runtime errors like the 2026-04-23 Gemma3
# `KeyError: None` in `ROPE_INIT_FUNCTIONS[self.rope_type]`.
#
# Phase B in the orchestrator skips the build entirely when the image
# tag is already in GAR, so the pull cost only hits when we actually
# rebuild — which is exactly when we want a fresh base.
echo ""
echo "Building image..."
docker build \
    --pull \
    -t "${FULL_IMAGE}" \
    -f "${REPO_ROOT}/gke/Dockerfile" \
    "${REPO_ROOT}"

# Push
echo ""
echo "Running: docker push ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo ""
echo "============================================================"
echo "  Done!"
echo "  Image: ${FULL_IMAGE}"
echo "============================================================"
