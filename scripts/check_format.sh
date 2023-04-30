#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <clang-format> <source_files>"
  exit 1
fi

CLANG_FORMAT=$1
shift
SOURCE_FILES=$@

for file in $SOURCE_FILES; do
  DIFF=$(diff -u <($CLANG_FORMAT -style=LLVM $file) $file) || true
  if [ ! -z "$DIFF" ]; then
    echo "Incorrect formatting in $file:"
    echo "$DIFF"
    exit 1
  fi
done