param([string]$Owned)
$ErrorActionPreference = 'Stop'
$targetIds = @($Owned.Split(',') | ForEach-Object { [int]$_ })
$dwmIds = @(Get-Process -Name dwm -ErrorAction Stop).Id
while ($true) {
    $samples = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
    $total = 0
    $valid = $true
    $found = @()
    $adapters = @()
    foreach ($sample in $samples) {
        if ($sample.InstanceName -match '^pid_(\d+)_(.*)$') {
            if ($targetIds -contains [int]$matches[1]) {
                $found += [int]$matches[1]
                if ($sample.Status -in @(0, 1)) {
                    $total += $sample.CookedValue
                    if ($sample.CookedValue -gt 0) { $adapters += $matches[2] }
                } else {
                    $valid = $false
                }
            }
        }
    }
    $dwm = 0
    foreach ($sample in $samples) {
        if ($sample.InstanceName -match '^pid_(\d+)_(.*)$') {
            if ([int]$matches[1] -in $dwmIds -and $matches[2] -in $adapters -and $sample.Status -in @(0, 1)) {
                $dwm = [Math]::Max([double]$dwm, [double]$sample.CookedValue)
            }
        }
    }
    @{ bytes = [long]$total; dwm_bytes = [long]$dwm; targets = $targetIds; found = $found; valid = $valid } | ConvertTo-Json -Compress
    Start-Sleep -Milliseconds 500
}
