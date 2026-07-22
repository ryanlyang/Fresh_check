param(
    [string]$Output = "teacher_logit_reco/presentation_assets/hlt_v2_strength_2p5_simple.jpg"
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
$grey = [System.Drawing.Color]::FromArgb(158, 169, 181)

$graphics.Clear($bg)

$fontTitle = New-Object System.Drawing.Font("Arial", 38, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Regular)
$fontLabel = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Regular)
$fontDrop = New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Bold)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushBlue = New-Object System.Drawing.SolidBrush($blue)
$brushBlueSoft = New-Object System.Drawing.SolidBrush($blueSoft)
$brushOrange = New-Object System.Drawing.SolidBrush($orange)
$brushOrangeSoft = New-Object System.Drawing.SolidBrush($orangeSoft)
$brushPanel = New-Object System.Drawing.SolidBrush($panel)
$brushGold = New-Object System.Drawing.SolidBrush($gold)
$brushGrey = New-Object System.Drawing.SolidBrush($grey)

$penInk = New-Object System.Drawing.Pen($ink, 2.0)
$penMuted = New-Object System.Drawing.Pen($muted, 2.5)
$penBlue = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(118, 166, 229), 2.0)
$penOrange = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(236, 135, 95), 2.0)
$penGold = New-Object System.Drawing.Pen($gold, 2.0)
$penGreyDashed = New-Object System.Drawing.Pen($grey, 2.0)
$penGreyDashed.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

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
    param($G, [double]$X, [double]$Y, [double]$R, $Brush, $Pen, [double]$Alpha = 1.0)
    if ($Alpha -lt 1.0) {
        $c = $Brush.Color
        $localBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb([int](255 * $Alpha), $c.R, $c.G, $c.B))
        $G.FillEllipse($localBrush, $X - $R, $Y - $R, $R * 2, $R * 2)
        $localBrush.Dispose()
    } else {
        $G.FillEllipse($Brush, $X - $R, $Y - $R, $R * 2, $R * 2)
    }
    $G.DrawEllipse($Pen, $X - $R, $Y - $R, $R * 2, $R * 2)
}

function Get-Particles {
    param([int]$Seed, [float]$Cx, [float]$Cy)
    $rng = [System.Random]::new($Seed)
    $particles = New-Object System.Collections.ArrayList
    for ($i = 0; $i -lt 58; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.2)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (26 + $rng.NextDouble() * 132)
        $y = $Cy + [Math]::Sin($theta) * $r * (20 + $rng.NextDouble() * 82)
        $pt = 2.5 + $rng.NextDouble() * 13.5
        [void]$particles.Add([pscustomobject]@{
            I = $i
            X = $x
            Y = $y
            Pt = $pt
            R = [Math]::Max(4.0, [Math]::Min(13.5, $pt * 0.82))
        })
    }
    return $particles
}

function Draw-Jet {
    param($G, $Particles, [float]$Cx, [float]$Cy, [bool]$AfterDrop)
    if ($AfterDrop) {
        $G.FillEllipse($brushOrangeSoft, $Cx - 160, $Cy - 105, 320, 210)
        $G.DrawEllipse($penOrange, $Cx - 160, $Cy - 105, 320, 210)
    } else {
        $G.FillEllipse($brushBlueSoft, $Cx - 160, $Cy - 105, 320, 210)
        $G.DrawEllipse($penBlue, $Cx - 160, $Cy - 105, 320, 210)
    }
    foreach ($p in $Particles) {
        Draw-Particle $G $p.X $p.Y $p.R $(if ($AfterDrop) { $brushOrange } else { $brushBlue }) $penInk 1.0
    }
}

function Draw-DroppedCloud {
    param($G, $Particles, [string]$Label, [float]$Cx, [float]$Cy, $Brush)
    $G.FillEllipse($brushPanel, $Cx - 78, $Cy - 58, 156, 116)
    $G.DrawEllipse($penGold, $Cx - 78, $Cy - 58, 156, 116)
    foreach ($p in $Particles) {
        $rx = $Cx + ($p.X % 78) - 39
        $ry = $Cy + ($p.Y % 54) - 27
        Draw-Particle $G $rx $ry ([Math]::Max(3.5, $p.R * 0.72)) $Brush $penGreyDashed 0.45
    }
    $size = $G.MeasureString($Label, $fontDrop)
    $G.DrawString($Label, $fontDrop, $brushInk, $Cx - $size.Width / 2, $Cy + 78)
}

$offlineParticles = Get-Particles 42 315 430
$kept = New-Object System.Collections.ArrayList
$thresholdDropped = New-Object System.Collections.ArrayList
$effDropped = New-Object System.Collections.ArrayList

foreach ($p in $offlineParticles) {
    if ($p.Pt -lt 5.1 -or $p.I % 17 -eq 0) {
        [void]$thresholdDropped.Add($p)
    } elseif ($p.I % 6 -eq 0 -or $p.I % 13 -eq 0) {
        [void]$effDropped.Add($p)
    } else {
        [void]$kept.Add([pscustomobject]@{
            I = $p.I
            X = 1285 + (($p.X - 315) * 0.94)
            Y = 430 + (($p.Y - 430) * 0.94)
            Pt = $p.Pt
            R = $p.R
        })
    }
}

$graphics.DrawString("How constituents get dropped", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("pseudo-HLT v2, strength 2.5: thresholding and efficiency loss", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 70 185 490 480 28 $brushPanel $penBlue
Draw-RoundedRect $graphics 1040 185 490 480 28 $brushPanel $penOrange

$graphics.DrawString("Offline jet", $fontLabel, $brushInk, 260, 217)
$graphics.DrawString("After drops", $fontLabel, $brushInk, 1237, 217)

Draw-Jet $graphics $offlineParticles 315 430 $false
Draw-Jet $graphics $kept 1285 430 $true

Draw-Arrow $graphics 560 430 665 430 $penMuted
Draw-Arrow $graphics 920 430 1035 430 $penMuted

Draw-DroppedCloud $graphics $thresholdDropped "threshold" 725 335 $brushBlue
Draw-DroppedCloud $graphics $effDropped "efficiency loss" 875 530 $brushOrange

Draw-Arrow $graphics 650 410 695 360 $penGreyDashed
Draw-Arrow $graphics 945 455 900 500 $penGreyDashed

$graphics.DrawString("same jet, fewer observed constituents", $fontSubtitle, $brushInk, 606, 808)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
