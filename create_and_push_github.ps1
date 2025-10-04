<#
Creates a GitHub repository using the GitHub REST API and pushes the current local repository to it.
Requires a Personal Access Token (PAT) with the "repo" scope.

Usage:
  1. Open PowerShell in the project directory.
  2. Run: .\create_and_push_github.ps1
  3. Enter your GitHub username, repository name (default will be 'materials-discovery-main'), and your PAT when prompted.

This script will:
  - Create the repository under your account (if it doesn't exist).
  - Set the remote URL to include the token for the push (temporary).
  - Push the current branch to GitHub (main branch by default).
  - Reset the remote URL to the non-token HTTPS form after pushing.

Security note: The PAT is only used in-memory and in the temporary push URL; it is not logged to disk. However, the command-line push using an embedded token may be visible in shell history. Use with care.
#>

# Get current directory and branch
$cwd = Get-Location
$branch = git branch --show-current
if (-not $branch) { $branch = "main" }

Write-Host "This will create a GitHub repo and push the current branch ('$branch')."

# Prompt for inputs
$username = Read-Host "GitHub username (owner)"
$repoNameInput = Read-Host "Repository name (press Enter to use 'materials-discovery-main')"
if ([string]::IsNullOrWhiteSpace($repoNameInput)) { $repo = "materials-discovery-main" } else { $repo = $repoNameInput }
$description = Read-Host "Repository description (optional)"

Write-Host "Enter a Personal Access Token (PAT) with 'repo' scope. It will not be echoed."
$secureToken = Read-Host -AsSecureString "PAT"
# Convert secure string to plain for API use (only in memory)
$ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
$token = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) | Out-Null

# Create the repository via GitHub API
$body = @{ name = $repo; description = $description; private = $false }
$headers = @{ Authorization = "token $token"; "User-Agent" = "$username-ps-script" }
$createUri = "https://api.github.com/user/repos"

Write-Host "Creating repository '$repo' under user '$username'..."
try {
    $resp = Invoke-RestMethod -Method Post -Uri $createUri -Headers $headers -Body ($body | ConvertTo-Json)
    Write-Host "Repository created: $($resp.html_url)"
} catch {
    $err = $_.Exception.Response
    if ($err -ne $null) {
        $reader = New-Object System.IO.StreamReader($err.GetResponseStream())
        $msg = $reader.ReadToEnd() | ConvertFrom-Json -ErrorAction SilentlyContinue
        Write-Host "GitHub API error: $msg.message"
        if ($msg.errors) { Write-Host ($msg.errors | Out-String) }
        Write-Host "If the repository already exists under your account, the script will continue and try to push."
    } else {
        Write-Host "Error creating repository: $_"
        throw $_
    }
}
# Validate the PAT by fetching the authenticated user (helps avoid username/token mismatch)
Write-Host "Validating token and resolving authenticated GitHub user..."
try {
    $userResp = Invoke-RestMethod -Method Get -Uri "https://api.github.com/user" -Headers $headers
    $apiLogin = $userResp.login
    Write-Host "Token authenticates as GitHub user: $apiLogin"
    if ($apiLogin -ne $username) {
        Write-Host "Warning: the username you entered ('$username') does not match the PAT owner ('$apiLogin'). Using '$apiLogin' as the owner."
        $username = $apiLogin
    }
} catch {
    Write-Host "Warning: unable to validate token via API. The token may be invalid or network access blocked. Error: $_"
    Write-Host "Proceeding, but repository creation or push may fail."
}

# Prepare remote push URL with token (temporary)
# URL-encode the token to avoid characters (like @ or :) breaking the URL
$encToken = [System.Uri]::EscapeDataString($token)
# Build push URL using format operator to avoid invalid variable parsing like $username:$encToken
$pushUrlWithToken = "https://{0}:{1}@github.com/{0}/{2}.git" -f $username, $encToken, $repo
$pushUrlNoToken = "https://github.com/$username/$repo.git"

# Ensure git remote 'origin' exists; if not, add it
$remotes = git remote
if ($remotes -notlike '*origin*') {
    git remote add origin $pushUrlNoToken
} else {
    git remote set-url origin $pushUrlNoToken
}

# Push using embedded token
Write-Host "Pushing branch '$branch' to GitHub (this will use the provided PAT temporarily)..."
try {
    # Use the push URL with embedded token directly to avoid storing token in remote config
    $pushCommand = "git push $pushUrlWithToken $branch:main -u"
    Write-Host "Running: git push <masked-url> $branch:main -u"
    Invoke-Expression $pushCommand
    Write-Host "Push complete."
} catch {
    Write-Host "Push failed: $_"
    Write-Host "You may need to verify your PAT scopes or GitHub permissions."
    throw $_
}

# Reset remote to non-token HTTPS URL
git remote set-url origin $pushUrlNoToken
Write-Host "Remote 'origin' set to $pushUrlNoToken (token removed)."

Write-Host "Done. You can visit: https://github.com/$username/$repo"
Write-Host "If you prefer SSH for future pushes, set the remote to the SSH URL: git remote set-url origin git@github.com:$username/$repo.git"

# Zero out the token variable
$token = $null

# End of script
