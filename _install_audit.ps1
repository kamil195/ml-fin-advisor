$sw = [System.Diagnostics.Stopwatch]::StartNew()
& .venv-audit\Scripts\python -m pip install --only-binary :all: -r _requirements-audit.txt >> _pip_audit.log 2>&1
$rc = $LASTEXITCODE
$sw.Stop()
"INSTALL_TIME_TOTAL:$($sw.Elapsed)" | Out-File -Append -Encoding ascii _pip_audit.log
"INSTALL_TIME_SECONDS:$($sw.Elapsed.TotalSeconds)" | Out-File -Append -Encoding ascii _pip_audit.log
"INSTALL_EXIT_CODE:$rc" | Out-File -Append -Encoding ascii _pip_audit.log
if ($rc -eq 0) { "INSTALL_STATUS:SUCCESS" } else { "INSTALL_STATUS:FAIL" }

