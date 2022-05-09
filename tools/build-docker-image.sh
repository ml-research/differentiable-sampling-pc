#!/usr/bin/env bash
set -euo pipefail

# Get project name
project_name="$(cat ./PROJECT_NAME)"

# Build docker image with project name as image name
docker build -t $project_name .
