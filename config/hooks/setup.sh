#!/usr/bin/env bash
set -e
git config core.hooksPath config/hooks
echo "Git hooks configured (core.hooksPath=config/hooks)."