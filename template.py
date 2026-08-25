"""Create the iFormat AI API package structure without overwriting code.

Run this module once with ``python template.py`` when bootstrapping a fresh
checkout. Re-running it is safe: existing files are never truncated.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PACKAGE_DIRECTORIES = (
    "app/core",
    "app/api/v1/endpoints",
    "app/schemas",
    "app/services",
    "app/utils",
    "app/tests",
    "app/data/iformat_kb",
)

PROJECT_FILES = (
    "app/__init__.py",
    "app/main.py",
    "app/core/__init__.py",
    "app/core/config.py",
    "app/core/exceptions.py",
    "app/api/__init__.py",
    "app/api/dependencies.py",
    "app/api/v1/__init__.py",
    "app/api/v1/api.py",
    "app/api/v1/endpoints/__init__.py",
    "app/api/v1/endpoints/ai.py",
    "app/schemas/__init__.py",
    "app/schemas/ai_schemas.py",
    "app/services/__init__.py",
    "app/services/bedrock_service.py",
    "app/services/rag_service.py",
    "app/utils/__init__.py",
    "app/utils/prompts.py",
    "app/tests/__init__.py",
    "app/tests/test_ai_endpoints.py",
    "app/data/iformat_kb/.gitkeep",
)


def create_project_structure() -> None:
    """Create all package directories and missing package marker files."""

    for relative_directory in PACKAGE_DIRECTORIES:
        (PROJECT_ROOT / relative_directory).mkdir(parents=True, exist_ok=True)

    for relative_file in PROJECT_FILES:
        (PROJECT_ROOT / relative_file).touch(exist_ok=True)


if __name__ == "__main__":
    create_project_structure()
    print("iFormat AI API structure is ready.")
