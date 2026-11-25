# -*- coding: utf-8 -*-
"""
HWIHWA LAB Universal Hugging Face Deployment Pipeline
Automated Multi-Target Deployment for ALOHA 14-DOF Bimanual Toss & Catch:
1) Hugging Face Spaces (Live 60fps Cyber HUD Telemetry Cockpit - Docker SDK)
2) Hugging Face Model Hub (Model Card, Weights, Telemetry & Full Production Bundle)

Usage:
  python deploy_to_hf.py --space     # Deploy live web cockpit to Hugging Face Spaces
  python deploy_to_hf.py --model     # Deploy weights and report to Hugging Face Model Hub
  python deploy_to_hf.py --all       # Deploy both Spaces and Model Hub sequentially
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DEFAULT_USERNAME = "hwihwalab"
DEFAULT_SPACE_REPO = "aloha-bimanual-toss-catch"
DEFAULT_MODEL_REPO = "aloha-bimanual-toss-catch-model"

SPACES_YAML_HEADER = """---
title: ALOHA 14-DOF Toss & Catch Cockpit
emoji: 🦾
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: mit
---

"""

IGNORE_PATTERNS = [
    "__pycache__/*",
    "*.pyc",
    ".git/*",
    ".gitignore",
    ".venv/*",
    "venv/*",
    "env/*",
    "*.log",
    ".system_generated/*",
    ".cursor/*",
    ".cursorrules*",
    ".agents/*",
    "DOCS_*",
    "docs/*",
    "*.egg-info/*",
    "MEMORY.md",
    "scratch/*",
    ".vscode/*",
    "*.tmp",
    "_TEMP_*",
    "checkpoints/*_steps.zip",
    "checkpoints/*_backup.zip"
]


def deploy_spaces(api: HfApi, username: str, repo_name: str, root_dir: Path):
    repo_id = f"{username}/{repo_name}"
    print("\n" + "=" * 65)
    print(f"🚀 [Hugging Face Spaces] Deploying Live Cyber Cockpit: {repo_id}")
    print("=" * 65)

    # 1. Create or ensure Space repository with static SDK (100% Free, Zero Cold Start)
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True
        )
        print(f"[1/4] Space repository initialized/confirmed: {url}")
    except Exception as e:
        print(f"[!] Note on Space repo setup: {e}")

    # 2. Prepare Spaces README with static frontmatter
    readme_path = root_dir / "README.md"
    temp_readme = root_dir / "_TEMP_SPACES_README.md"
    if readme_path.exists():
        raw_text = readme_path.read_text(encoding="utf-8")
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                raw_text = parts[2].strip()
        temp_readme.write_text(SPACES_YAML_HEADER + raw_text, encoding="utf-8")
        print("[2/4] Prepared Spaces README with Static SDK frontmatter.")

    # 3. Upload core static web cockpit files via upload_folder with strict ignore patterns
    print("[3/4] Uploading 60fps Cyber Cockpit web engine and assets to Spaces...")
    spaces_ignore = IGNORE_PATTERNS + [
        "aloha_bimanual_toss_catch_bundle.zip",
        "*.py",
        "Dockerfile",
        ".dockerignore",
        "checkpoints/*",
        "envs/*",
        "rewards/*",
        "render/*",
        "build_release.ps1"
    ]
    api.upload_folder(
        folder_path=str(root_dir),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=spaces_ignore,
        commit_message="feat: deploy 60fps Live Cyber Cockpit on Static SDK"
    )

    # 4. Upload the generated Spaces README as README.md
    if temp_readme.exists():
        print("[4/4] Finalizing Spaces README.md...")
        api.upload_file(
            path_or_fileobj=str(temp_readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            commit_message="docs: set Spaces static metadata & live cockpit docs"
        )
        temp_readme.unlink()

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n" + "-" * 65)
    print(f"🎉 Spaces Deployment Successfully Pushed!")
    print(f"👉 Live App URL: {space_url}")
    print("-----------------------------------------------------------------\n")


def deploy_model_hub(api: HfApi, username: str, repo_name: str, root_dir: Path):
    repo_id = f"{username}/{repo_name}"
    print("\n" + "=" * 65)
    print(f"🚀 [Hugging Face Model Hub] Deploying Weights & Model Card: {repo_id}")
    print("=" * 65)

    # 1. Create or ensure Model repository
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True
        )
        print(f"[1/3] Model repository initialized/confirmed: {url}")
    except Exception as e:
        print(f"[!] Note on Model repo setup: {e}")

    # 2. Upload Model Hub folder with strict ignore patterns
    print("[2/3] Uploading 9 Whitelist Artifacts (Weights, Bundle, Assets, README)...")
    # For model hub, we ignore Dockerfile, dockerignore, run_live_cockpit.py to keep model hub clean
    model_ignore = IGNORE_PATTERNS + [
        "Dockerfile",
        ".dockerignore",
        "deploy_to_hf.py",
        "build_release.ps1",
        "setup_static_cockpit.py",
        "extract_cockpit_trajectories.py"
    ]
    api.upload_folder(
        folder_path=str(root_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=model_ignore,
        commit_message="feat: upload ALOHA 14-DOF PPO final weights, bundle & evaluation report"
    )

    model_url = f"https://huggingface.co/{repo_id}"
    print("\n" + "-" * 65)
    print(f"🎉 Model Hub Deployment Successfully Pushed!")
    print(f"👉 Model Hub URL: {model_url}")
    print("-----------------------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="HWIHWA LAB Hugging Face Deployment Pipeline")
    parser.add_argument("--space", action="store_true", help="Deploy live web cockpit to Hugging Face Spaces")
    parser.add_argument("--model", action="store_true", help="Deploy weights & report to Hugging Face Model Hub")
    parser.add_argument("--all", action="store_true", help="Deploy both Spaces and Model Hub sequentially")
    parser.add_argument("--username", type=str, default=DEFAULT_USERNAME, help="Hugging Face username or org")
    parser.add_argument("--space-repo", type=str, default=DEFAULT_SPACE_REPO, help="Spaces repository name")
    parser.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO, help="Model Hub repository name")
    args = parser.parse_args()

    if not args.space and not args.model and not args.all:
        parser.print_help()
        print("\n[!] Please specify --space, --model, or --all.")
        sys.exit(1)

    root_dir = Path(__file__).resolve().parent

    print("\n[HWIHWA LAB] Authenticating with Hugging Face Hub...")
    api = HfApi()
    user_info = api.whoami()
    current_user = user_info["name"]
    print(f"✓ Authenticated as: {current_user}")

    target_user = args.username or current_user

    if args.space or args.all:
        deploy_spaces(api, target_user, args.space_repo, root_dir)

    if args.model or args.all:
        deploy_model_hub(api, target_user, args.model_repo, root_dir)


if __name__ == "__main__":
    main()
