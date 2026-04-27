#!/usr/bin/env python3
"""
DockGraph Repository Analysis and Cleanup Tool

This script helps you:
1. Analyze current repository structure
2. Identify files to keep/remove
3. Verify before pushing to GitHub
4. Generate cleanup recommendations

Usage:
    python repo_manager.py analyze
    python repo_manager.py cleanup --dry-run
    python repo_manager.py cleanup
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess


class DockGraphRepoManager:
    """Manage DockGraph repository structure and cleanup."""
    
    def __init__(self, repo_path: str = None):
        """Initialize with repository path."""
        if repo_path is None:
            # Auto-detect if running in repo
            repo_path = os.path.expanduser("~/Protein Docking/DockGraph")
        
        self.repo_path = Path(repo_path)
        if not self.repo_path.exists():
            # Try relative path
            self.repo_path = Path.cwd()
        
        self.git_path = self.repo_path / ".git"
        self.config = {
            "keep_dirs": [
                "scripts",
                "src",
                "data/benchmark",
                "experiments/20260307_073713_full_training",
                "docs",
                ".github",
            ],
            "keep_files": [
                "README.md",
                "requirements.txt",
                "setup.py",
                ".gitignore",
                "LICENSE",
                ".gitattributes",
            ],
            "remove_patterns": [
                "old_*.py",
                "backup_*.py",
                "deprecated_*.py",
                "test_*.py",  # Only if in wrong location
                "*.ipynb",  # Optional: remove notebooks
                "*.tmp",
                "*.swp",
                "*~",
                ".DS_Store",
            ],
            "remove_dirs": [
                "__pycache__",
                ".ipynb_checkpoints",
                ".vscode",
                ".idea",
                ".pytest_cache",
                ".mypy_cache",
                "venv",
                "env",
                ".venv",
            ],
        }
    
    def analyze(self) -> Dict:
        """Analyze repository structure."""
        analysis = {
            "total_files": 0,
            "total_dirs": 0,
            "python_files": [],
            "large_files": [],
            "deprecated_files": [],
            "cache_dirs": [],
            "total_size": 0,
            "git_status": {},
        }
        
        print(f"Analyzing repository: {self.repo_path}")
        print("=" * 70)
        
        # Walk directory tree
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            dirs[:] = [d for d in dirs if d != ".git"]
            
            for d in dirs:
                analysis["total_dirs"] += 1
                dir_path = os.path.join(root, d)
                if d in self.config["remove_dirs"]:
                    analysis["cache_dirs"].append(dir_path)
            
            for f in files:
                analysis["total_files"] += 1
                file_path = os.path.join(root, f)
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                # Check file size
                try:
                    size = os.path.getsize(file_path)
                    analysis["total_size"] += size
                    
                    if size > 50 * 1024 * 1024:  # > 50MB
                        analysis["large_files"].append(
                            (rel_path, size / (1024 * 1024))
                        )
                except OSError:
                    pass
                
                # Check Python files
                if f.endswith(".py"):
                    analysis["python_files"].append(rel_path)
                
                # Check deprecated files
                for pattern in self.config["remove_patterns"]:
                    if f.endswith(pattern.lstrip("*")) or f.startswith(
                        pattern.rstrip("*")
                    ):
                        analysis["deprecated_files"].append(rel_path)
        
        return analysis
    
    def print_analysis(self, analysis: Dict) -> None:
        """Print analysis results."""
        print("\n📊 REPOSITORY ANALYSIS")
        print("=" * 70)
        
        print(f"\n📁 Structure:")
        print(f"  Total Files: {analysis['total_files']}")
        print(f"  Total Directories: {analysis['total_dirs']}")
        print(f"  Total Size: {analysis['total_size'] / (1024**3):.2f} GB")
        
        print(f"\n🐍 Python Files ({len(analysis['python_files'])}):")
        for pf in sorted(analysis["python_files"])[:15]:
            print(f"  - {pf}")
        if len(analysis["python_files"]) > 15:
            print(f"  ... and {len(analysis['python_files']) - 15} more")
        
        if analysis["large_files"]:
            print(f"\n📦 Large Files (>50MB):")
            for lf, size in sorted(
                analysis["large_files"], key=lambda x: x[1], reverse=True
            ):
                print(f"  - {lf}: {size:.2f} MB")
        
        if analysis["deprecated_files"]:
            print(f"\n🗑️  Deprecated Files ({len(analysis['deprecated_files'])}):")
            for df in analysis["deprecated_files"]:
                print(f"  - {df}")
        
        if analysis["cache_dirs"]:
            print(f"\n🧹 Cache Directories to Remove ({len(analysis['cache_dirs'])}):")
            for cd in analysis["cache_dirs"]:
                print(f"  - {cd}")
    
    def get_git_status(self) -> Dict:
        """Get git status if in a repo."""
        if not self.git_path.exists():
            return {"is_git_repo": False}
        
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo_path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            return {
                "is_git_repo": True,
                "modified": len(
                    [
                        l
                        for l in result.stdout.split("\n")
                        if l.startswith("M ")
                    ]
                ),
                "added": len(
                    [l for l in result.stdout.split("\n") if l.startswith("A ")]
                ),
                "deleted": len(
                    [l for l in result.stdout.split("\n") if l.startswith("D ")]
                ),
            }
        except Exception as e:
            return {"is_git_repo": True, "error": str(e)}
    
    def cleanup(self, dry_run: bool = True) -> None:
        """Remove unnecessary files."""
        print(f"\n🧹 CLEANUP (dry_run={dry_run})")
        print("=" * 70)
        
        removed_count = 0
        
        # Remove cache directories
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d != ".git"]
            
            for d in dirs:
                dir_path = Path(root) / d
                if d in self.config["remove_dirs"]:
                    if dry_run:
                        print(f"  [DRY RUN] Would remove: {dir_path}")
                    else:
                        import shutil
                        shutil.rmtree(dir_path)
                        print(f"  ✓ Removed: {dir_path}")
                    removed_count += 1
            
            # Remove deprecated files
            for f in files:
                file_path = Path(root) / f
                for pattern in self.config["remove_patterns"]:
                    if (f.endswith(pattern.lstrip("*")) 
                        or f.startswith(pattern.rstrip("*"))):
                        if dry_run:
                            print(f"  [DRY RUN] Would remove: {file_path}")
                        else:
                            file_path.unlink()
                            print(f"  ✓ Removed: {file_path}")
                        removed_count += 1
                        break
        
        print(f"\n{removed_count} items would be removed (dry_run={dry_run})")
    
    def generate_report(self) -> str:
        """Generate a detailed report."""
        analysis = self.analyze()
        git_status = self.get_git_status()
        
        report = f"""
# DockGraph Repository Analysis Report
Generated: {Path.cwd()}

## Summary
- Total Files: {analysis['total_files']}
- Total Directories: {analysis['total_dirs']}
- Python Scripts: {len(analysis['python_files'])}
- Deprecated Files: {len(analysis['deprecated_files'])}
- Cache Directories: {len(analysis['cache_dirs'])}
- Total Size: {analysis['total_size'] / (1024**3):.2f} GB

## Git Status
- Is Git Repo: {git_status.get('is_git_repo', False)}
- Modified Files: {git_status.get('modified', 0)}
- Added Files: {git_status.get('added', 0)}
- Deleted Files: {git_status.get('deleted', 0)}

## Recommended Actions

### 1. Remove Cache Directories
```bash
find . -type d -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {{}} +
rm -rf .vscode .idea .pytest_cache
```

### 2. Remove Deprecated Files
```bash
rm -f scripts/old_*.py scripts/backup_*.py scripts/deprecated_*.py
```

### 3. Update .gitignore
See GITHUB_UPDATE_GUIDE.md for complete .gitignore template

### 4. Commit and Push
```bash
git add -A
git commit -m "refactor: Clean repository structure"
git push origin main
```

## Python Files Found
{json.dumps(analysis['python_files'], indent=2)}

## Large Files (>50MB)
{json.dumps([(f, f"{s:.2f}MB") for f, s in analysis['large_files']], indent=2)}
"""
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DockGraph Repository Management Tool"
    )
    parser.add_argument(
        "action",
        choices=["analyze", "cleanup", "report"],
        help="Action to perform",
    )
    parser.add_argument(
        "--repo",
        default=os.path.expanduser("~/Protein Docking/DockGraph"),
        help="Repository path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be removed without actually removing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually remove files (without --dry-run flag)",
    )
    
    args = parser.parse_args()
    
    # Expand path
    repo_path = os.path.expanduser(args.repo)
    
    if not os.path.exists(repo_path):
        print(f"❌ Repository not found: {repo_path}")
        sys.exit(1)
    
    manager = DockGraphRepoManager(repo_path)
    
    if args.action == "analyze":
        analysis = manager.analyze()
        manager.print_analysis(analysis)
        git_status = manager.get_git_status()
        print(f"\n🔗 Git Status: {git_status}")
    
    elif args.action == "cleanup":
        dry_run = not args.force
        manager.cleanup(dry_run=dry_run)
        if dry_run:
            print("\n💡 Run with --force to actually remove files")
    
    elif args.action == "report":
        report = manager.generate_report()
        print(report)
        
        # Save to file
        report_path = Path(repo_path) / "REPO_ANALYSIS_REPORT.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n✓ Report saved to: {report_path}")


if __name__ == "__main__":
    main()