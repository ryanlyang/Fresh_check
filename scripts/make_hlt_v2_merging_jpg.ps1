param(
    [string]$Output = "teacher_logit_reco/presentation_assets/hlt_v2_merging_simple.jpg"
)

Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$outPath = Join-Path $root $Output
$outDir = Split-Path -Parent $outPath
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$width = 1600
$height = 900
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$bg = [System.Drawing.Color]::FromArgb(248, 245, 238)
$ink = [System.Drawing.Color]::FromArgb(24, 35, 46)
$muted = [System.Drawing.Color]::FromArgb(84, 99, 115)
$blue = [System.Drawing.Color]::FromArgb(47, 128, 237)
$blueSoft = [System.Drawing.Color]::FromArgb(214, 232, 252)
$orange = [System.Drawing.Color]::FromArgb(228, 87, 46)
$orangeSoft = [System.Drawing.Color]::FromArgb(250, 224, 213)
$gold = [System.Drawing.Color]::FromArgb(202, 146, 54)
$panel = [System.Drawing.Color]::FromArgb(255, 255, 255)

$graphics.Clear($bg)

$fontTitle = New-Object System.Drawing.Font("Arial", 38, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Regular)
$fontLabel = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Bold)
$fontMid = New-Object System.Drawing.Font("Arial", 17, [System.Drawing.FontStyle]::Bold)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushBlue = New-Object System.Drawing.SolidBrush($blue)
$brushBlueSoft = New-Object System.Drawing.SolidBrush($blueSoft)
$brushOrange = New-Object System.Drawing.SolidBrush($orange)
$brushOrangeSoft = New-Object System.Drawing.SolidBrush($orangeSoft)
$brushPanel = New-Object System.Drawing.SolidBrush($panel)
$brushGold = New-Object System.Drawing.SolidBrush($gold)

$penInk = New-Object System.Drawing.Pen($ink, 2.0)
$penMuted = New-Object System.Drawing.Pen($muted, 2.5)
$penBlue = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(118, 166, 229), 2.0)
$penOrange = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(236, 135, 95), 2.0)
$penGold = New-Object System.Drawing.Pen($gold, 2.0)
$penGoldDashed = New-Object System.Drawing.Pen($gold, 2.0)
$penGoldDashed.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

function Draw-RoundedRect {
    param($G, [float]$X, [float]$Y, [float]$W, [float]$H, [float]$R, $Brush, $Pen)
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $R * 2
    $path.AddArc($X, $Y, $d, $d, 180, 90)
    $path.AddArc($X + $W - $d, $Y, $d, $d, 270, 90)
    $path.AddArc($X + $W - $d, $Y + $H - $d, $d, $d, 0, 90)
    $path.AddArc($X, $Y + $H - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    if ($Brush -ne $null) { $G.FillPath($Brush, $path) }
    if ($Pen -ne $null) { $G.DrawPath($Pen, $path) }
    $path.Dispose()
}

function Draw-Arrow {
    param($G, [float]$X1, [float]$Y1, [float]$X2, [float]$Y2, $Pen)
    $localPen = New-Object System.Drawing.Pen($Pen.Color, $Pen.Width)
    $localPen.DashStyle = $Pen.DashStyle
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(7, 9)
    $localPen.CustomEndCap = $cap
    $G.DrawLine($localPen, $X1, $Y1, $X2, $Y2)
    $localPen.Dispose()
    $cap.Dispose()
}

function Draw-Particle {
    param($G, [double]$X, [double]$Y, [double]$R, $Brush, $Pen)
    $G.FillEllipse($Brush, $X - $R, $Y - $R, $R * 2, $R * 2)
    $G.DrawEllipse($Pen, $X - $R, $Y - $R, $R * 2, $R * 2)
}

function Get-Particles {
    param([int]$Seed, [float]$Cx, [float]$Cy)
    $rng = [System.Random]::new($Seed)
    $particles = New-Object System.Collections.ArrayList
    for ($i = 0; $i -lt 50; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (30 + $rng.NextDouble() * 128)
        $y = $Cy + [Math]::Sin($theta) * $r * (22 + $rng.NextDouble() * 84)
        $pt = 3.5 + $rng.NextDouble() * 13.0
        [void]$particles.Add([pscustomobject]@{
            I = $i
            X = $x
            Y = $y
            Pt = $pt
            R = [Math]::Max(4.0, [Math]::Min(13.5, $pt * 0.8))
        })
    }
    [void]$particles.Add([pscustomobject]@{ I = 100; X = $Cx - 26; Y = $Cy - 24; Pt = 12.0; R = 12.0 })
    [void]$particles.Add([pscustomobject]@{ I = 101; X = $Cx - 6; Y = $Cy - 12; Pt = 8.0; R = 9.0 })
    [void]$particles.Add([pscustomobject]@{ I = 102; X = $Cx + 34; Y = $Cy + 35; Pt = 10.0; R = 10.5 })
    [void]$particles.Add([pscustomobject]@{ I = 103; X = $Cx + 52; Y = $Cy + 44; Pt = 6.5; R = 8.0 })
    return $particles
}

function Draw-Jet {
    param($G, $Particles, [float]$Cx, [float]$Cy, [bool]$Merged)
    if ($Merged) {
        $G.FillEllipse($brushOrangeSoft, $Cx - 160, $Cy - 105, 320, 210)
        $G.DrawEllipse($penOrange, $Cx - 160, $Cy - 105, 320, 210)
    } else {
        $G.FillEllipse($brushBlueSoft, $Cx - 160, $Cy - 105, 320, 210)
        $G.DrawEllipse($penBlue, $Cx - 160, $Cy - 105, 320, 210)
    }
    foreach ($p in $Particles) {
        Draw-Particle $G $p.X $p.Y $p.R $(if ($Merged) { $brushOrange } else { $brushBlue }) $penInk
    }
}

$offlineParticles = Get-Particles 87 315 430
$mergedParticles = New-Object System.Collections.ArrayList
foreach ($p in $offlineParticles) {
    if ($p.I -eq 101 -or $p.I -eq 103) {
        continue
    }
    if ($p.I -eq 100) {
        [void]$mergedParticles.Add([pscustomobject]@{ I = 200; X = 1285 - 17; Y = 430 - 18; Pt = 20.0; R = 17.0 })
    } elseif ($p.I -eq 102) {
        [void]$mergedParticles.Add([pscustomobject]@{ I = 201; X = 1285 + 40; Y = 430 + 40; Pt = 16.5; R = 15.0 })
    } else {
        [void]$mergedParticles.Add([pscustomobject]@{
            I = $p.I
            X = 1285 + (($p.X - 315) * 0.94)
            Y = 430 + (($p.Y - 430) * 0.94)
            Pt = $p.Pt
            R = $p.R
        })
    }
}

$graphics.DrawString("How nearby constituents merge", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("pseudo-HLT v2: close particles can become one observed token", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 70 185 490 480 28 $brushPanel $penBlue
Draw-RoundedRect $graphics 1040 185 490 480 28 $brushPanel $penOrange

$graphics.DrawString("Offline jet", $fontLabel, $brushInk, 260, 217)
$graphics.DrawString("After merging", $fontLabel, $brushInk, 1215, 217)

Draw-Jet $graphics $offlineParticles 315 430 $false
Draw-Jet $graphics $mergedParticles 1285 430 $true

Draw-Arrow $graphics 560 430 655 430 $penMuted
Draw-Arrow $graphics 945 430 1035 430 $penMuted

$graphics.FillEllipse($brushPanel, 665, 275, 275, 245)
$graphics.DrawEllipse($penGold, 665, 275, 275, 245)
$graphics.DrawEllipse($penGoldDashed, 705, 322, 92, 72)
$graphics.DrawEllipse($penGoldDashed, 810, 395, 82, 64)
Draw-Particle $graphics 737 355 22 $brushBlue $penInk
Draw-Particle $graphics 775 366 17 $brushBlue $penInk
Draw-Arrow $graphics 797 360 842 360 $penMuted
Draw-Particle $graphics 874 360 31 $brushOrange $penInk

Draw-Particle $graphics 828 428 20 $brushBlue $penInk
Draw-Particle $graphics 865 438 15 $brushBlue $penInk
Draw-Arrow $graphics 892 434 930 434 $penMuted
Draw-Particle $graphics 952 434 28 $brushOrange $penInk

$mergeText = "merge"
$mergeSize = $graphics.MeasureString($mergeText, $fontMid)
$graphics.DrawString($mergeText, $fontMid, $brushInk, 802 - $mergeSize.Width / 2, 548)

$graphics.DrawString("close pair", $fontMid, $brushInk, 682, 625)
$graphics.DrawString("merged token", $fontMid, $brushInk, 842, 625)
$graphics.DrawString("same jet, lower constituent granularity", $fontSubtitle, $brushInk, 585, 808)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
