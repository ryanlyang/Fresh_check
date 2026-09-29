param(
    [string]$RemoteHost = 'ryreu@tigris.rc.rit.edu',
    [string]$RemoteDirectory = '/home/ryreu/atlas/jetclass_test20m',
    [string]$Archive = 'C:\Users\22rya\Downloads\JetClass_Pythia_test_20M.tar'
)
$ErrorActionPreference = 'Stop'
if ($RemoteHost -notmatch '^[a-zA-Z0-9_.@-]+$' -or $RemoteDirectory -notmatch '^/[a-zA-Z0-9_./-]+$') {
    throw 'Use a simple absolute remote directory and user@host.'
}
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "Missing archive: $Archive" }
$archiveBytes = (Get-Item -LiteralPath $Archive).Length
# Peak storage is archive + extracted ROOT files + ~10GB predictions + headroom.
$requiredBytes = 2 * $archiveBytes + 15GB
& ssh $RemoteHost "mkdir -p -- '$RemoteDirectory' && df -B1 '$RemoteDirectory' && test ! -e '$RemoteDirectory/JetClass_Pythia_test_20M.tar' && test ! -e '$RemoteDirectory/test_20M' && test `"`$(df -B1 --output=avail '$RemoteDirectory' | tail -1 | tr -d ' ')`" -ge $requiredBytes"
if ($LASTEXITCODE -ne 0) { throw 'Remote preflight failed: existing upload/extraction or insufficient space. Nothing was overwritten.' }
Write-Host 'Uploading the 30.4GB archive. Duo authentication is expected.'
& scp $Archive "${RemoteHost}:${RemoteDirectory}/JetClass_Pythia_test_20M.tar"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed. The original local archive is unchanged; a partial remote upload may remain.' }
Write-Host 'Computing local SHA-256 for transfer verification...'
$archiveHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
& ssh $RemoteHost "printf '%s  %s\n' '$archiveHash' '$RemoteDirectory/JetClass_Pythia_test_20M.tar' | sha256sum -c -"
if ($LASTEXITCODE -ne 0) { throw 'Remote archive checksum did not match. Do not extract or evaluate it.' }
Write-Host "Upload verified. On Tigris run: tar --keep-old-files -xf '$RemoteDirectory/JetClass_Pythia_test_20M.tar' -C '$RemoteDirectory'"
Write-Host 'The upload does not extract data, remove files, or submit jobs.'
