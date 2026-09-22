#!/usr/bin/env bash
# One-click live proxy. Double-click in Finder or run: bash play.sh
cd "$(dirname "$0")"
exec /Users/guangyi.liu/miniconda3/bin/python3 examples/play.py --video "$@"
