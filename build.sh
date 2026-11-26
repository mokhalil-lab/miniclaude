#!/bin/sh
# Build the agent binary.
#
#   ./build.sh          -> ./miniclaude (this directory)
#   go install .        -> $(go env GOPATH)/bin/miniclaude  (run from ANY directory)
#
# After `go install`, make sure $(go env GOPATH)/bin is on your PATH, then run:
#   miniclaude -p "what files are in this directory?"
set -e
cd "$(dirname "$0")"
go build -o miniclaude .
echo "built ./miniclaude"
