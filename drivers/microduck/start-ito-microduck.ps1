$ErrorActionPreference='Stop'
$base=$PSScriptRoot
try {
    $exe=Join-Path $base 'dist\ito\ito.exe'
    if (!(Test-Path $exe)) {throw 'Ito is missing. Restore the installed Ito folder.'}
    $ip=((wsl.exe -d ito-microduck -u root -- hostname -I).Trim() -split '\s+')[0]
    function Ready {
        $client=New-Object System.Net.Sockets.TcpClient
        try { $task=$client.ConnectAsync($ip,8081); return ($task.Wait(500) -and $client.Connected) } catch {return $false} finally {$client.Dispose()}
    }
    if (!(Ready)) {
        Start-Process -FilePath (Join-Path $base 'microduck-sim.cmd') -WorkingDirectory $base
        Write-Host 'Starting virtual Microduck and its viewer...'
        $deadline=(Get-Date).AddSeconds(90)
        while (!(Ready)) {
            if ((Get-Date) -gt $deadline) {throw 'Microduck did not start. Check its simulator window for the error.'}
            Start-Sleep -Milliseconds 300
        }
    }
    $code=(wsl.exe -d ito-microduck -u root -- cat /root/.config/ito/microduck-sim-code).Trim()
    Start-Process -FilePath $exe -ArgumentList @("${ip}:8081",'--code',$code) -WorkingDirectory $base
    Write-Host 'Ito is open. Close the Microduck viewer to stop the simulator.'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    Read-Host 'Press Enter to close'
    exit 1
}
