# Step 1: Check for existing SSH key
$sshKeyPath = "$HOME\.ssh\id_rsa"
if (Test-Path $sshKeyPath) {
    Write-Host "SSH key already exists at $sshKeyPath."
} else {
    # Step 2: Generate a new SSH key
    Write-Host "No SSH key found. Generating a new SSH key..."
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f $sshKeyPath -N ""
    Write-Host "SSH key generated at $sshKeyPath."
}

# Step 3: Add the SSH key to the SSH agent
Write-Host "Starting the SSH agent and adding the SSH key..."
Start-Service ssh-agent -ErrorAction SilentlyContinue
ssh-add $sshKeyPath

# Step 4: Provide instructions to add the SSH key to GitHub
Write-Host "Copying the SSH public key to the clipboard..."
Get-Content "$sshKeyPath.pub" | clip
Write-Host "The SSH public key has been copied to your clipboard."
Write-Host "Please add it to your GitHub account at https://github.com/settings/keys."
Write-Host "Press Enter after adding the SSH key to continue..."
Read-Host

# Step 5: Set the Git remote to the SSH URL
$repoUrl = "git@github.com:thedevz43/materials-discovery-main.git"
Write-Host "Setting the Git remote to $repoUrl..."
git remote set-url origin $repoUrl

# Step 6: Test the SSH connection
Write-Host "Testing the SSH connection to GitHub..."
ssh -T git@github.com

# Step 7: Push the project to GitHub
Write-Host "Pushing the project to GitHub..."
git push -u origin main