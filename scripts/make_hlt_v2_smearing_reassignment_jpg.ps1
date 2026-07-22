param(
    [string]$Output = "teacher_logit_reco/presentation_assets/hlt_v2_smearing_reassignment_simple.jpg"
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
$fontMid = New-Object System.Drawing.Font("Arial", 17, [System.Drawing.FontStyle]::Bold)

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
    for ($i = 0; $i -lt 48; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (28 + $rng.NextDouble() * 132)
        $y = $Cy + [Math]::Sin($theta) * $r * (20 + $rng.NextDouble() * 86)
        $pt = 3.5 + $rng.NextDouble() * 13.0
        [void]$particles.Add([pscustomobject]@{
            I = $i
            X = $x
            Y = $y
            Pt = $pt
            R = [Math]::Max(4.0, [Math]::Min(13.5, $pt * 0.8))
        })
    }
    return $particles
}

function Draw-Jet {
    param($G, $Particles, [float]$Cx, [float]$Cy, [bool]$Shifted)
    if ($Shifted) {
        $G.FillEllipse($brushOrangeSoft, $Cx - 160, $Cy - 105, 320, 210)
        $G.DrawEllipse($penOrange, $Cx - 160, $Cy - 105, 320, 210)
    } else {
        $G.FillEllipse($brushBlueSoft, $Cx - 160, $Cy - 105, 320, 210)
        $G.DrawEllipse($penBlue, $Cx - 160, $Cy - 105, 320, 210)
    }
    foreach ($p in $Particles) {
        Draw-Particle $G $p.X $p.Y $p.R $(if ($Shifted) { $brushOrange } else { $brushBlue }) $penInk 1.0
    }
}

$offlineParticles = Get-Particles 118 315 430
$shiftedParticles = New-Object System.Collections.ArrayList
$rng = [System.Random]::new(51)
foreach ($p in $offlineParticles) {
    $dx = ($rng.NextDouble() - 0.5) * 28
    $dy = ($rng.NextDouble() - 0.5) * 22
    $scale = 0.72 + $rng.NextDouble() * 0.58
    [void]$shiftedParticles.Add([pscustomobject]@{
        I = $p.I
        X = 1285 + (($p.X - 315) * 0.94) + $dx
        Y = 430 + (($p.Y - 430) * 0.94) + $dy
        Pt = $p.Pt * $scale
        R = [Math]::Max(4.0, [Math]::Min(15.0, $p.R * $scale))
    })
}

$graphics.DrawString("How constituents get smeared", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("pseudo-HLT v2: observed particles keep their identity, but measured pT and direction shift", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 70 185 490 480 28 $brushPanel $penBlue
Draw-RoundedRect $graphics 1040 185 490 480 28 $brushPanel $penOrange

$graphics.DrawString("Offline jet", $fontLabel, $brushInk, 260, 217)
$graphics.DrawString("After smearing", $fontLabel, $brushInk, 1208, 217)

Draw-Jet $graphics $offlineParticles 315 430 $false
Draw-Jet $graphics $shiftedParticles 1285 430 $true

Draw-Arrow $graphics 560 430 655 430 $penMuted
Draw-Arrow $graphics 945 430 1035 430 $penMuted

Draw-RoundedRect $graphics 650 285 310 245 28 $brushPanel $penGold
$graphics.DrawEllipse($penGoldDashed, 700, 330, 205, 135)

Draw-Particle $graphics 735 365 20 $brushBlue $penInk 0.58
Draw-Arrow $graphics 735 365 790 338 $penGreyDashed
Draw-Particle $graphics 790 338 16 $brushOrange $penInk 1.0

Draw-Particle $graphics 760 450 13 $brushBlue $penInk 0.55
Draw-Arrow $graphics 760 450 820 468 $penGreyDashed
Draw-Particle $graphics 820 468 18 $brushOrange $penInk 1.0

Draw-Particle $graphics 858 385 18 $brushBlue $penInk 0.50
Draw-Arrow $graphics 858 385 830 420 $penGreyDashed
Draw-Particle $graphics 830 420 13 $brushOrange $penInk 1.0

$graphics.DrawString("smearing", $fontMid, $brushInk, 748, 562)
$graphics.DrawString("same observed constituents, shifted measurements", $fontSubtitle, $brushInk, 535, 808)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
