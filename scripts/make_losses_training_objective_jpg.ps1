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
$green = [System.Drawing.Color]::FromArgb(36, 148, 99)
$greenSoft = [System.Drawing.Color]::FromArgb(220, 242, 231)
$blue = [System.Drawing.Color]::FromArgb(47, 128, 237)
$blueSoft = [System.Drawing.Color]::FromArgb(214, 232, 252)
$purple = [System.Drawing.Color]::FromArgb(116, 91, 184)
$purpleSoft = [System.Drawing.Color]::FromArgb(234, 229, 248)
$gold = [System.Drawing.Color]::FromArgb(202, 146, 54)
$goldSoft = [System.Drawing.Color]::FromArgb(254, 244, 216)
$orange = [System.Drawing.Color]::FromArgb(228, 87, 46)
$orangeSoft = [System.Drawing.Color]::FromArgb(250, 224, 213)
$panel = [System.Drawing.Color]::FromArgb(255, 255, 255)

$graphics.Clear($bg)

$fontTitle = New-Object System.Drawing.Font("Arial", 38, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Regular)
$fontLabel = New-Object System.Drawing.Font("Arial", 20, [System.Drawing.FontStyle]::Bold)
$fontMid = New-Object System.Drawing.Font("Arial", 17, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Regular)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushGreen = New-Object System.Drawing.SolidBrush($green)
$brushGreenSoft = New-Object System.Drawing.SolidBrush($greenSoft)
$brushBlue = New-Object System.Drawing.SolidBrush($blue)
$brushBlueSoft = New-Object System.Drawing.SolidBrush($blueSoft)
$brushPurple = New-Object System.Drawing.SolidBrush($purple)
$brushPurpleSoft = New-Object System.Drawing.SolidBrush($purpleSoft)
$brushGold = New-Object System.Drawing.SolidBrush($gold)
$brushGoldSoft = New-Object System.Drawing.SolidBrush($goldSoft)
$brushOrange = New-Object System.Drawing.SolidBrush($orange)
$brushOrangeSoft = New-Object System.Drawing.SolidBrush($orangeSoft)
$brushPanel = New-Object System.Drawing.SolidBrush($panel)

$penInk = New-Object System.Drawing.Pen($ink, 2.0)
$penMuted = New-Object System.Drawing.Pen($muted, 2.5)
$penGreen = New-Object System.Drawing.Pen($green, 2.2)
$penBlue = New-Object System.Drawing.Pen($blue, 2.2)
$penPurple = New-Object System.Drawing.Pen($purple, 2.2)
$penGold = New-Object System.Drawing.Pen($gold, 2.2)
$penOrange = New-Object System.Drawing.Pen($orange, 2.2)
$penDashed = New-Object System.Drawing.Pen($muted, 2.0)
$penDashed.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

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
    param($G, [int]$Seed, [float]$Cx, [float]$Cy, $ParticleBrush, $FillBrush, $Pen, [int]$N, [double]$Scale)
    $rng = [System.Random]::new($Seed)
    $G.FillEllipse($FillBrush, $Cx - 115, $Cy - 72, 230, 144)
    $G.DrawEllipse($Pen, $Cx - 115, $Cy - 72, 230, 144)
    for ($i = 0; $i -lt $N; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (16 + $rng.NextDouble() * 90)
        $y = $Cy + [Math]::Sin($theta) * $r * (12 + $rng.NextDouble() * 55)
        $pt = (3.5 + $rng.NextDouble() * 11.0) * $Scale
        $rr = [Math]::Max(3.5, [Math]::Min(11.0, $pt * 0.72))
        Draw-Particle $G $x $y $rr $ParticleBrush $penInk
    }
}

function Draw-LossBox {
    param($G, [float]$X, [float]$Y, [string]$Title, [string]$Sub, $Brush, $Pen)
    Draw-RoundedRect $G $X $Y 330 125 24 $Brush $Pen
    Center-Text $G $Title $fontLabel $brushInk ($X + 165) ($Y + 29)
    Center-Text $G $Sub $fontSmall $brushMuted ($X + 165) ($Y + 68)
}

$graphics.DrawString("Losses / training objective", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("train corrected soft views with three reconstruction terms", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 85 245 310 250 26 $brushPanel $penGreen
Center-Text $graphics "Corrected view" $fontLabel $brushInk 240 272
Draw-Jet $graphics 29 240 390 $brushGreen $brushGreenSoft $penGreen 42 1.0

Draw-RoundedRect $graphics 85 565 310 210 26 $brushPanel $penBlue
Center-Text $graphics "Offline target" $fontLabel $brushInk 240 592
Draw-Jet $graphics 43 240 695 $brushBlue $brushBlueSoft $penBlue 45 1.03

Draw-Arrow $graphics 395 370 555 305 $penMuted
Draw-Arrow $graphics 395 390 555 430 $penMuted
Draw-Arrow $graphics 395 695 555 555 $penMuted

Draw-LossBox $graphics 555 210 "Set matching loss" "particle set alignment" $brushGoldSoft $penGold
Draw-LossBox $graphics 555 385 "Generation loss" "added-particle quality" $brushPurpleSoft $penPurple
Draw-LossBox $graphics 555 560 "Jet pT loss" "global pT scale" $brushOrangeSoft $penOrange

Draw-RoundedRect $graphics 1015 312 400 250 30 $brushPanel $penMuted
Center-Text $graphics "Total objective" $fontLabel $brushInk 1215 346
Center-Text $graphics "set matching" $fontMid $brushGold 1215 404
Center-Text $graphics "+ generation" $fontMid $brushPurple 1215 444
Center-Text $graphics "+ jet pT" $fontMid $brushOrange 1215 484

Draw-Arrow $graphics 885 272 1015 390 $penMuted
Draw-Arrow $graphics 885 447 1015 430 $penMuted
Draw-Arrow $graphics 885 622 1015 470 $penMuted

$graphics.DrawString("goal: corrected views look offline-like while preserving jet-scale pT", $fontSubtitle, $brushInk, 420, 832)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
