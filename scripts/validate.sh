#!/bin/bash
# 本地验证脚本 - 运行与CI相同的检查

set -e  # 遇到错误立即退出

echo "🔍 Running code quality checks..."

echo "📋 Checking code format with ruff..."
uv run ruff format --check src tests config

echo "🔎 Running linter with ruff..."
uv run ruff check src tests config

echo "🧪 Running tests with pytest..."
uv run pytest tests -v

echo "✅ All checks passed!"