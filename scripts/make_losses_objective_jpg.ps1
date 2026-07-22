param(
    [string]$Output = "teacher_logit_reco/presentation_assets/losses_training_objective_simple.jpg"
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
$orange = [System.Drawing.Color]::FromArgb(228, 87, 46)
$orangeSoft = [System.Drawing.Color]::FromArgb(250, 224, 213)
$blue = [System.Drawing.Color]::FromArgb(47, 128, 237)
$blueSoft = [System.Drawing.Color]::FromArgb(214, 232, 252)
$green = [System.Drawing.Color]::FromArgb(36, 148, 99)
$greenSoft = [System.Drawing.Color]::FromArgb(220, 242, 231)
$teal = [System.Drawing.Color]::FromArgb(20, 145, 151)
$tealSoft = [System.Drawing.Color]::FromArgb(216, 242, 243)
$purple = [System.Drawing.Color]::FromArgb(116, 91, 184)
$purpleSoft = [System.Drawing.Color]::FromArgb(234, 229, 248)
$gold = [System.Drawing.Color]::FromArgb(202, 146, 54)
$goldSoft = [System.Drawing.Color]::FromArgb(254, 244, 216)
$panel = [System.Drawing.Color]::FromArgb(255, 255, 255)

$graphics.Clear($bg)

$fontTitle = New-Object System.Drawing.Font("Arial", 38, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Regular)
$fontLabel = New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Bold)
$fontMid = New-Object System.Drawing.Font("Arial", 17, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Regular)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushOrange = New-Object System.Drawing.SolidBrush($orange)
$brushOrangeSoft = New-Object System.Drawing.SolidBrush($orangeSoft)
$brushBlue = New-Object System.Drawing.SolidBrush($blue)
$brushBlueSoft = New-Object System.Drawing.SolidBrush($blueSoft)
$brushGreen = New-Object System.Drawing.SolidBrush($green)
$brushGreenSoft = New-Object System.Drawing.SolidBrush($greenSoft)
$brushTeal = New-Object System.Drawing.SolidBrush($teal)
$brushTealSoft = New-Object System.Drawing.SolidBrush($tealSoft)
$brushPurple = New-Object System.Drawing.SolidBrush($purple)
$brushPurpleSoft = New-Object System.Drawing.SolidBrush($purpleSoft)
$brushGold = New-Object System.Drawing.SolidBrush($gold)
$brushGoldSoft = New-Object System.Drawing.SolidBrush($goldSoft)
$brushPanel = New-Object System.Drawing.SolidBrush($panel)

$penInk = New-Object System.Drawing.Pen($ink, 2.0)
$penMuted = New-Object System.Drawing.Pen($muted, 2.5)
$penOrange = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(236, 135, 95), 2.0)
$penBlue = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(118, 166, 229), 2.0)
$penGreen = New-Object System.Drawing.Pen($green, 2.2)
$penTeal = New-Object System.Drawing.Pen($teal, 2.2)
$penPurple = New-Object System.Drawing.Pen($purple, 2.2)
$penGold = New-Object System.Drawing.Pen($gold, 2.2)
$penGoldDashed = New-Object System.Drawing.Pen($gold, 2.2)
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

function Center-Text {
    param($G, [string]$Text, $Font, $Brush, [float]$Cx, [float]$Y)
    $size = $G.MeasureString($Text, $Font)
    $G.DrawString($Text, $Font, $Brush, $Cx - $size.Width / 2, $Y)
}

function Draw-Particle {
    param($G, [double]$X, [double]$Y, [double]$R, $Brush, $Pen)
    $G.FillEllipse($Brush, $X - $R, $Y - $R, $R * 2, $R * 2)
    $G.DrawEllipse($Pen, $X - $R, $Y - $R, $R * 2, $R * 2)
}

function Draw-Jet {
    param($G, [int]$Seed, [float]$Cx, [float]$Cy, $ParticleBrush, $FillBrush, $Pen, [int]$N, [double]$Scale = 1.0)
    $rng = [System.Random]::new($Seed)
    $G.FillEllipse($FillBrush, $Cx - 108, $Cy - 68, 216, 136)
    $G.DrawEllipse($Pen, $Cx - 108, $Cy - 68, 216, 136)
    for ($i = 0; $i -lt $N; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (16 + $rng.NextDouble() * 84)
        $y = $Cy + [Math]::Sin($theta) * $r * (12 + $rng.NextDouble() * 52)
        $pt = (3.5 + $rng.NextDouble() * 11.0) * $Scale
        $rr = [Math]::Max(3.3, [Math]::Min(10.5, $pt * 0.73))
        Draw-Particle $G $x $y $rr $ParticleBrush $penInk
    }
}

function Draw-LossCard {
    param($G, [float]$X, [float]$Y, [string]$Title, [string]$Sub, $Brush, $Pen)
    Draw-RoundedRect $G $X $Y 360 150 24 $Brush $Pen
    Center-Text $G $Title $fontLabel $brushInk ($X + 180) ($Y + 28)
    Center-Text $G $Sub $fontSmall $brushMuted ($X + 180) ($Y + 70)
}

$graphics.DrawString("Losses / training objective", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("train corrected soft views to match the offline jet using three reconstruction terms", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 95 245 330 240 26 $brushPanel $penGreen
Center-Text $graphics "Corrected views" $fontLabel $brushInk 260 275
Draw-Jet $graphics 33 260 382 $brushGreen $brushGreenSoft $penGreen 40 1.0

Draw-RoundedRect $graphics 95 560 330 220 26 $brushPanel $penBlue
Center-Text $graphics "Offline target" $fontLabel $brushInk 260 590
Draw-Jet $graphics 44 260 690 $brushBlue $brushBlueSoft $penBlue 48 1.05

Draw-Arrow $graphics 425 372 620 255 $penMuted
Draw-Arrow $graphics 425 680 620 255 $penMuted
Draw-Arrow $graphics 425 372 620 455 $penMuted
Draw-Arrow $graphics 425 680 620 455 $penMuted
Draw-Arrow $graphics 425 372 620 655 $penMuted
Draw-Arrow $graphics 425 680 620 655 $penMuted

Draw-LossCard $graphics 620 180 "Set matching loss" "match predicted particles to offline particles" $brushGoldSoft $penGold
Draw-LossCard $graphics 620 390 "Generation loss" "make added particles useful, not random" $brushTealSoft $penTeal
Draw-LossCard $graphics 620 600 "Jet pT loss" "match total jet transverse momentum" $brushPurpleSoft $penPurple

Draw-RoundedRect $graphics 1115 315 365 225 28 $brushPanel $penGold
Center-Text $graphics "Total objective" $fontLabel $brushInk 1298 350
$graphics.DrawString("L =", $fontTitle, $brushInk, 1160, 420)
$graphics.DrawString("set", $fontMid, $brushGold, 1238, 408)
$graphics.DrawString("+ gen", $fontMid, $brushTeal, 1286, 408)
$graphics.DrawString("+ jet pT", $fontMid, $brushPurple, 1355, 408)
Center-Text $graphics "only these three terms" $fontSmall $brushMuted 1298 486

Draw-Arrow $graphics 980 255 1115 405 $penGold
Draw-Arrow $graphics 980 465 1115 430 $penTeal
Draw-Arrow $graphics 980 675 1115 455 $penPurple

$graphics.DrawString("particle-level", $fontSmall, $brushMuted, 692, 350)
$graphics.DrawString("candidate-level", $fontSmall, $brushMuted, 684, 560)
$graphics.DrawString("jet-level", $fontSmall, $brushMuted, 710, 770)

$graphics.DrawString("goal: corrected views look offline-like while preserving the jet-scale pT", $fontSubtitle, $brushInk, 400, 835)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
