param(
    [string]$Output = "teacher_logit_reco/presentation_assets/m2_losses_training_objective.jpg"
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
$green = [System.Drawing.Color]::FromArgb(39, 151, 111)
$greenSoft = [System.Drawing.Color]::FromArgb(218, 241, 233)
$gold = [System.Drawing.Color]::FromArgb(202, 146, 54)
$goldSoft = [System.Drawing.Color]::FromArgb(254, 244, 216)
$purple = [System.Drawing.Color]::FromArgb(116, 91, 184)
$purpleSoft = [System.Drawing.Color]::FromArgb(234, 229, 248)
$panel = [System.Drawing.Color]::FromArgb(255, 255, 255)

$graphics.Clear($bg)

$fontTitle = New-Object System.Drawing.Font("Arial", 38, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Regular)
$fontLabel = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Bold)
$fontMid = New-Object System.Drawing.Font("Arial", 17, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Regular)
$fontEquation = New-Object System.Drawing.Font("Consolas", 22, [System.Drawing.FontStyle]::Bold)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushBlue = New-Object System.Drawing.SolidBrush($blue)
$brushBlueSoft = New-Object System.Drawing.SolidBrush($blueSoft)
$brushOrange = New-Object System.Drawing.SolidBrush($orange)
$brushOrangeSoft = New-Object System.Drawing.SolidBrush($orangeSoft)
$brushGreen = New-Object System.Drawing.SolidBrush($green)
$brushGreenSoft = New-Object System.Drawing.SolidBrush($greenSoft)
$brushGold = New-Object System.Drawing.SolidBrush($gold)
$brushGoldSoft = New-Object System.Drawing.SolidBrush($goldSoft)
$brushPurple = New-Object System.Drawing.SolidBrush($purple)
$brushPurpleSoft = New-Object System.Drawing.SolidBrush($purpleSoft)
$brushPanel = New-Object System.Drawing.SolidBrush($panel)

$penInk = New-Object System.Drawing.Pen($ink, 2.0)
$penMuted = New-Object System.Drawing.Pen($muted, 2.4)
$penBlue = New-Object System.Drawing.Pen($blue, 2.0)
$penOrange = New-Object System.Drawing.Pen($orange, 2.0)
$penGreen = New-Object System.Drawing.Pen($green, 2.0)
$penGold = New-Object System.Drawing.Pen($gold, 2.2)
$penPurple = New-Object System.Drawing.Pen($purple, 2.0)
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

function Draw-ParticleCloud {
    param($G, [int]$Seed, [float]$Cx, [float]$Cy, $MainBrush, $SoftBrush, $Pen, [int]$N)
    $rng = [System.Random]::new($Seed)
    $G.FillEllipse($SoftBrush, $Cx - 140, $Cy - 88, 280, 176)
    $G.DrawEllipse($Pen, $Cx - 140, $Cy - 88, 280, 176)
    for ($i = 0; $i -lt $N; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (18 + $rng.NextDouble() * 110)
        $y = $Cy + [Math]::Sin($theta) * $r * (14 + $rng.NextDouble() * 68)
        $pt = 3.0 + $rng.NextDouble() * 10.0
        $rr = [Math]::Max(3.5, [Math]::Min(10.5, $pt * 0.72))
        Draw-Particle $G $x $y $rr $MainBrush $penInk
    }
}

function Draw-ParticleCloudScaled {
    param($G, [int]$Seed, [float]$Cx, [float]$Cy, $MainBrush, $SoftBrush, $Pen, [int]$N, [float]$Scale)
    $rng = [System.Random]::new($Seed)
    $G.FillEllipse($SoftBrush, $Cx - 140 * $Scale, $Cy - 88 * $Scale, 280 * $Scale, 176 * $Scale)
    $G.DrawEllipse($Pen, $Cx - 140 * $Scale, $Cy - 88 * $Scale, 280 * $Scale, 176 * $Scale)
    for ($i = 0; $i -lt $N; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (18 + $rng.NextDouble() * 110) * $Scale
        $y = $Cy + [Math]::Sin($theta) * $r * (14 + $rng.NextDouble() * 68) * $Scale
        $pt = 3.0 + $rng.NextDouble() * 10.0
        $rr = [Math]::Max(3.0, [Math]::Min(9.0, $pt * 0.68 * $Scale))
        Draw-Particle $G $x $y $rr $MainBrush $penInk
    }
}

function Draw-LossLens {
    param($G, [float]$X, [float]$Y, [string]$Title, [string]$Sub, $Fill, $Pen, $TitleBrush)
    Draw-RoundedRect $G $X $Y 325 95 18 $Fill $Pen
    Center-Text $G $Title $fontMid $TitleBrush ($X + 162.5) ($Y + 18)
    Center-Text $G $Sub $fontSmall $brushMuted ($X + 162.5) ($Y + 52)
}

$graphics.DrawString("Losses / Training Objective", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("the offline jet is the target; the losses compare the reconstruction to it", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 80 250 310 245 26 $brushPanel $penBlue
Center-Text $graphics "Pseudo-HLT input" $fontLabel $brushInk 235 268
Draw-ParticleCloud $graphics 9 235 390 $brushBlue $brushBlueSoft $penBlue 27

Draw-RoundedRect $graphics 510 250 310 245 26 $brushGoldSoft $penGold
Center-Text $graphics "Reconstructor" $fontLabel $brushInk 665 268
Draw-RoundedRect $graphics 585 352 160 80 18 $brushPanel $penGold
Center-Text $graphics "predicts" $fontSmall $brushMuted 665 372
Center-Text $graphics "corrected set" $fontMid $brushInk 665 397

Draw-RoundedRect $graphics 900 250 310 245 26 $brushPanel $penGreen
Center-Text $graphics "Reconstructed view" $fontLabel $brushInk 1055 268
Draw-ParticleCloud $graphics 21 1055 390 $brushGreen $brushGreenSoft $penGreen 36

Draw-RoundedRect $graphics 1275 210 255 325 30 $brushPanel $penOrange
Center-Text $graphics "Offline target" $fontLabel $brushInk 1402.5 248
Draw-ParticleCloudScaled $graphics 31 1402.5 390 $brushOrange $brushOrangeSoft $penOrange 44 0.80
Center-Text $graphics "matched same jet" $fontSmall $brushMuted 1402.5 508

Draw-Arrow $graphics 390 372 510 372 $penMuted
Draw-Arrow $graphics 820 372 900 372 $penMuted
Draw-Arrow $graphics 1210 372 1275 372 $penMuted

Draw-RoundedRect $graphics 535 575 675 135 24 $brushPanel $penPurple
Center-Text $graphics "Compare reconstructed view to offline target" $fontLabel $brushInk 872.5 603
Center-Text $graphics "all three terms ask: did the corrected HLT view become more offline-like?" $fontSmall $brushMuted 872.5 638

Draw-Arrow $graphics 1055 495 765 575 $penDashed
Draw-Arrow $graphics 1402.5 535 980 575 $penDashed

Draw-LossLens $graphics 105 735 "Set matching loss" "unordered particle set agreement" $brushPanel $penBlue $brushBlue
Draw-LossLens $graphics 638 735 "Generation loss" "recover missing constituents" $brushPanel $penGreen $brushGreen
Draw-LossLens $graphics 1170 735 "Jet pT loss" "match total offline jet pT" $brushPanel $penOrange $brushOrange

Draw-Arrow $graphics 705 708 345 735 $penDashed
Draw-Arrow $graphics 872 710 800 735 $penDashed
Draw-Arrow $graphics 1040 708 1332 735 $penDashed

$equation = "L = w_set L_set + w_gen L_gen + w_pT L_jet-pT"
Center-Text $graphics $equation $fontEquation $brushInk 800 842

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
