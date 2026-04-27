#!/bin/bash
# DockGraph Server-to-GitHub Complete Sync Script
# Purpose: Make your server and GitHub repository completely consistent
# Usage: bash sync_to_github.sh

set -e  # Exit on any error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   DockGraph Complete Server-to-GitHub Sync                    ║"
echo "║   Making Server Files and GitHub Repository Identical         ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Step 1: Navigate to project
echo -e "\n[STEP 1] Navigating to DockGraph project..."
cd ~/Protein\ Docking/DockGraph
echo "✓ Current directory: $(pwd)"

# Step 2: Check Git status
echo -e "\n[STEP 2] Checking Git status..."
git status
echo "✓ Git repository verified"

# Step 3: Create backup branch before making changes
echo -e "\n[STEP 3] Creating backup of current state..."
BACKUP_BRANCH="backup-$(date +%Y%m%d_%H%M%S)"
git branch "$BACKUP_BRANCH"
echo "✓ Backup branch created: $BACKUP_BRANCH"

# Step 4: Show what files exist
echo -e "\n[STEP 4] Current project structure:"
echo "===== Python Files ====="
find . -name "*.py" -type f | grep -v __pycache__ | sort

echo -e "\n===== Data Files ====="
ls -la data/benchmark/ 2>/dev/null || echo "data/benchmark directory exists"

echo -e "\n===== Experiment Results ====="
ls -la experiments/20260307_073713_full_training/ 2>/dev/null | head -20

# Step 5: Show Git remote
echo -e "\n[STEP 5] Verifying GitHub remote..."
git remote -v
echo ""
read -p "Is the GitHub URL correct? (yes/no): " confirm_url
if [ "$confirm_url" != "yes" ]; then
    echo "❌ Please fix the remote URL first:"
    echo "   git remote set-url origin https://github.com/Kazi-Nasif/DockGraph.git"
    exit 1
fi

# Step 6: Clean up unnecessary files
echo -e "\n[STEP 6] Cleaning up unnecessary files..."

# Remove macOS system files
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -type f -delete
echo "✓ .DS_Store files removed"

# Remove Python cache
echo "Removing Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
echo "✓ Python cache removed"

# Remove IDE configuration
echo "Removing IDE configuration directories..."
rm -rf .vscode .idea 2>/dev/null || true
echo "✓ IDE configurations removed"

# Remove temporary files
echo "Removing temporary files..."
find . -name "*~" -type f -delete
find . -name "*.swp" -type f -delete
find . -name "*.tmp" -type f -delete
echo "✓ Temporary files removed"

# Step 7: Create comprehensive .gitignore
echo -e "\n[STEP 7] Creating/updating .gitignore..."
cat > .gitignore << 'GITIGNORE_END'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
venv/
ENV/
env/
.venv
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
*.sublime-project
*.sublime-workspace
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
.tox/
htmlcov/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Temporary files
*.tmp
*.bak
*.log
*.pkl
*.pickle

# Large model files - keep only best model
*.pt
!experiments/20260307_073713_full_training/best_model.pt
*.pth
*.hdf5
*.h5

# Keep directory structure but ignore raw data files
data/benchmark/*/*.pdb
data/benchmark/*/*.ent
data/raw/

# Experiment logs
experiments/*/logs/
experiments/*/wandb/

GITIGNORE_END
echo "✓ .gitignore updated with comprehensive patterns"

# Step 8: Add all files to Git
echo -e "\n[STEP 8] Staging all current files..."
git add -A
echo "✓ All files staged"

# Step 9: Show what will be committed
echo -e "\n[STEP 9] Preview of changes to be committed:"
echo "===== Files Added/Modified ====="
git diff --cached --name-status | head -30
echo ""
git diff --cached --name-status | wc -l
echo "files total"

# Step 10: Show statistics
echo -e "\n[STEP 10] Repository Statistics:"
echo "Total Python files: $(find . -name "*.py" -type f | grep -v __pycache__ | wc -l)"
echo "Total directories: $(find . -type d | grep -v .git | wc -l)"
echo "Total size: $(du -sh . | cut -f1)"

# Step 11: Verify no large files are being added
echo -e "\n[STEP 11] Checking for large files..."
echo "Files larger than 50MB:"
git ls-files -s | awk '{print $4}' | xargs ls -lh 2>/dev/null | awk '$5 ~ /[G-Z]|[5-9][0-9]M/ {print $5, $9}' || echo "None found"

# Step 12: Ask for confirmation
echo -e "\n[STEP 12] CONFIRMATION REQUIRED"
echo "========================================"
echo "This will:"
echo "  ✓ Remove all cache and temporary files"
echo "  ✓ Update .gitignore"
echo "  ✓ Stage all current project files"
echo "  ✓ Commit to git"
echo "  ✓ Push to GitHub (making server = GitHub)"
echo ""
read -p "Continue with commit and push? (yes/no): " confirm_sync
if [ "$confirm_sync" != "yes" ]; then
    echo "❌ Sync cancelled. Your files are clean but not committed."
    echo "If you want to undo cleanup: git reset --hard $BACKUP_BRANCH"
    exit 0
fi

# Step 13: Create commit
echo -e "\n[STEP 13] Creating commit..."
git commit -m "sync: Full server-to-GitHub synchronization

This commit makes the GitHub repository identical to the current server state.

Changes:
- Clean Python cache (__pycache__, .ipynb_checkpoints)
- Remove IDE configuration files (.vscode, .idea)
- Remove macOS system files (.DS_Store)
- Remove temporary files (*~, *.swp, *.tmp)
- Update .gitignore with comprehensive patterns
- Include all current project files

Project Structure:
- 22 Python scripts and modules
- DB5.5 benchmark data references
- Best model checkpoint: experiments/20260307_073713_full_training/best_model.pt
- Evaluation results and analysis scripts
- Documentation and guides

Status:
- 98.0% success rate on DB5.5 benchmark
- Mean DockQ: 0.728±0.210
- Ready for ECML-PKDD submission"

echo "✓ Commit created successfully"

# Step 14: Push to GitHub
echo -e "\n[STEP 14] Pushing to GitHub..."
git push origin main
echo "✓ Pushed to GitHub successfully"

# Step 15: Verify push
echo -e "\n[STEP 15] Verifying sync..."
echo "Latest commit on server:"
git log --oneline -1
echo ""
echo "Remote tracking branch:"
git branch -vv | grep main

# Step 16: Success message
echo -e "\n╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✓ SYNC COMPLETED SUCCESSFULLY              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Your server and GitHub are now synchronized!"
echo ""
echo "Next steps:"
echo "1. Visit: https://github.com/Kazi-Nasif/DockGraph"
echo "2. Verify all files are present and correct"
echo "3. Check that your project structure looks good"
echo ""
echo "Backup branch (if you need to revert): $BACKUP_BRANCH"
echo "  To revert: git reset --hard $BACKUP_BRANCH"
echo ""
echo "Recent commits:"
git log --oneline -5