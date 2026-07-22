param(
    [string]$Output = "teacher_logit_reco/presentation_assets/inputs_outputs_simple.jpg"
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
$fontSmall = New-Object System.Drawing.Font("Arial", 15, [System.Drawing.FontStyle]::Regular)
$fontMid = New-Object System.Drawing.Font("Arial", 17, [System.Drawing.FontStyle]::Bold)
$fontTiny = New-Object System.Drawing.Font("Arial", 13, [System.Drawing.FontStyle]::Regular)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushOrange = New-Object System.Drawing.SolidBrush($orange)
$brushOrangeSoft = New-Object System.Drawing.SolidBrush($orangeSoft)
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
$penMuted = New-Object System.Drawing.Pen($muted, 2.7)
$penOrange = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(236, 135, 95), 2.0)
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

function Draw-Jet {
    param(
        $G,
        [int]$Seed,
        [float]$Cx,
        [float]$Cy,
        $ParticleBrush,
        $FillBrush,
        $Pen,
        [int]$N,
        [double]$Scale = 1.0,
        [double]$Alpha = 1.0
    )
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
        Draw-Particle $G $x $y $rr $ParticleBrush $penInk $Alpha
    }
}

function Center-Text {
    param($G, [string]$Text, $Font, $Brush, [float]$Cx, [float]$Y)
    $size = $G.MeasureString($Text, $Font)
    $G.DrawString($Text, $Font, $Brush, $Cx - $size.Width / 2, $Y)
}

$graphics.DrawString("Inputs and outputs", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("reconstructor-only view: one pseudo-HLT input, three soft corrected views", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 70 285 330 300 26 $brushPanel $penOrange
Center-Text $graphics "Pseudo-HLT jet" $fontLabel $brushInk 235 313
Draw-Jet $graphics 19 235 440 $brushOrange $brushOrangeSoft $penOrange 30 0.95
Center-Text $graphics "input" $fontSmall $brushMuted 235 540

Draw-RoundedRect $graphics 635 285 330 300 26 $brushGoldSoft $penGold
Center-Text $graphics "Reconstructor" $fontLabel $brushInk 800 318
Draw-RoundedRect $graphics 700 388 200 72 18 $brushPanel $penGold
Center-Text $graphics "m2_base" $fontMid $brushInk 800 407
Center-Text $graphics "HLT -> soft views" $fontSmall $brushMuted 800 435

Draw-RoundedRect $graphics 635 665 330 92 24 $brushPanel $penGoldDashed
Center-Text $graphics "Offline jet" $fontLabel $brushInk 800 685
Center-Text $graphics "training target only" $fontSmall $brushMuted 800 718

Draw-RoundedRect $graphics 1195 155 335 185 24 $brushPanel $penGreen
Center-Text $graphics "Corrected view 1" $fontLabel $brushInk 1362 168
Draw-Jet $graphics 29 1362 270 $brushGreen $brushGreenSoft $penGreen 42 0.92 0.90

Draw-RoundedRect $graphics 1195 360 335 185 24 $brushPanel $penTeal
Center-Text $graphics "Corrected view 2" $fontLabel $brushInk 1362 373
Draw-Jet $graphics 31 1362 475 $brushTeal $brushTealSoft $penTeal 39 0.90 0.88

Draw-RoundedRect $graphics 1195 565 335 185 24 $brushPanel $penPurple
Center-Text $graphics "Corrected view 3" $fontLabel $brushInk 1362 578
Draw-Jet $graphics 37 1362 680 $brushPurple $brushPurpleSoft $penPurple 45 0.88 0.88

Draw-Arrow $graphics 400 435 635 435 $penMuted
Draw-Arrow $graphics 965 385 1195 250 $penMuted
Draw-Arrow $graphics 965 435 1195 452 $penMuted
Draw-Arrow $graphics 965 485 1195 658 $penMuted
Draw-Arrow $graphics 800 665 800 585 $penGoldDashed

$graphics.DrawString("matched supervision", $fontTiny, $brushMuted, 832, 621)
$graphics.DrawString("soft alternatives", $fontTiny, $brushMuted, 1010, 392)

$graphics.DrawString("output is not a single hard correction: it is a small set of corrected views", $fontSubtitle, $brushInk, 430, 830)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
