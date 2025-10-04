# Automate pushing local repo to GitHub
import subprocess
import sys

repo_url = "https://github.com/thedevz43/materials-discovery.git"

# Add remote, set branch, and push
subprocess.run(["git", "remote", "add", "origin", repo_url])
subprocess.run(["git", "branch", "-M", "main"])
subprocess.run(["git", "push", "-u", "origin", "main"])

print(f"Project pushed to {repo_url}")
