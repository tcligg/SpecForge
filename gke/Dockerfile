# Dockerfile for SpecForge data regeneration on GKE.
#
# Copies the local SpecForge source tree into the image and installs it.
# Does NOT clone from GitHub — build context is the local repo.
#
# Build (from the SpecForge repo root):
#   docker build -t specforge-regen:latest -f gke/Dockerfile .
#
# The entrypoint script (regen_entrypoint.sh) will:
#   1. Launch SGLang server(s) for the target model.
#   2. Wait for server health.
#   3. Run regenerate_train_data.py with chunk-based processing.
#   4. Shut down servers on exit.

FROM lmsysorg/sglang:dev

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/specforge

# Copy dependency spec first for better layer caching.
# If only source code changes (not deps), pip install is cached.
COPY pyproject.toml version.txt ./
RUN pip install --no-cache-dir -e "." 2>&1 || true

# Copy the rest of the source tree
COPY . .

# Re-run install to pick up the actual package (the first run
# may partially fail without the source, but it caches deps).
RUN pip install --no-cache-dir -e "."
# Make sure we use latest transformers for Gemma4.
RUN pip install -U transformers

# Make entrypoint executable
RUN chmod +x /app/specforge/gke/regen_entrypoint.sh

ENTRYPOINT ["/app/specforge/gke/regen_entrypoint.sh"]
