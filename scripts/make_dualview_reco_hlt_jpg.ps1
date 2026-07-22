param(
    [string]$Output = "teacher_logit_reco/presentation_assets/dualview_reco_hlt_simple.jpg"
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
$gold = [System.Drawing.Color]::FromArgb(202, 146, 54)
$goldSoft = [System.Drawing.Color]::FromArgb(254, 244, 216)
$blue = [System.Drawing.Color]::FromArgb(47, 128, 237)
$blueSoft = [System.Drawing.Color]::FromArgb(214, 232, 252)
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
$brushGreen = New-Object System.Drawing.SolidBrush($green)
$brushGreenSoft = New-Object System.Drawing.SolidBrush($greenSoft)
$brushGold = New-Object System.Drawing.SolidBrush($gold)
$brushGoldSoft = New-Object System.Drawing.SolidBrush($goldSoft)
$brushBlue = New-Object System.Drawing.SolidBrush($blue)
$brushBlueSoft = New-Object System.Drawing.SolidBrush($blueSoft)
$brushPanel = New-Object System.Drawing.SolidBrush($panel)

$penInk = New-Object System.Drawing.Pen($ink, 2.0)
$penMuted = New-Object System.Drawing.Pen($muted, 2.5)
$penOrange = New-Object System.Drawing.Pen($orange, 2.2)
$penGreen = New-Object System.Drawing.Pen($green, 2.2)
$penGold = New-Object System.Drawing.Pen($gold, 2.2)
$penBlue = New-Object System.Drawing.Pen($blue, 2.2)

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
    $G.FillEllipse($FillBrush, $Cx - 118, $Cy - 76, 236, 152)
    $G.DrawEllipse($Pen, $Cx - 118, $Cy - 76, 236, 152)
    for ($i = 0; $i -lt $N; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (16 + $rng.NextDouble() * 88)
        $y = $Cy + [Math]::Sin($theta) * $r * (12 + $rng.NextDouble() * 54)
        $pt = (3.5 + $rng.NextDouble() * 11.0) * $Scale
        $rr = [Math]::Max(3.3, [Math]::Min(10.5, $pt * 0.73))
        Draw-Particle $G $x $y $rr $ParticleBrush $penInk
    }
}

function Draw-Encoder {
    param($G, [float]$X, [float]$Y, [string]$Title, [string]$Sub, $Brush, $Pen)
    Draw-RoundedRect $G $X $Y 270 130 24 $Brush $Pen
    Center-Text $G $Title $fontLabel $brushInk ($X + 135) ($Y + 31)
    Center-Text $G $Sub $fontSmall $brushMuted ($X + 135) ($Y + 72)
}

$graphics.DrawString("Dual-view tagger", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("base HLT view and reconstructed view are encoded separately, then fused", $fontSubtitle, $brushMuted, 75, 111)

Draw-RoundedRect $graphics 80 205 330 245 26 $brushPanel $penOrange
Center-Text $graphics "Base HLT view" $fontLabel $brushInk 245 232
Draw-Jet $graphics 19 245 342 $brushOrange $brushOrangeSoft $penOrange 30 0.95

Draw-RoundedRect $graphics 80 525 330 245 26 $brushPanel $penGreen
Center-Text $graphics "Reconstructed view" $fontLabel $brushInk 245 552
Draw-Jet $graphics 29 245 662 $brushGreen $brushGreenSoft $penGreen 42 1.05

Draw-Encoder $graphics 575 230 "HLT encoder" "Particle Transformer branch" $brushOrangeSoft $penOrange
Draw-Encoder $graphics 575 550 "Reco encoder" "Particle Transformer branch" $brushGreenSoft $penGreen

Draw-RoundedRect $graphics 965 355 265 190 30 $brushGoldSoft $penGold
Center-Text $graphics "Fusion head" $fontLabel $brushInk 1098 392
Center-Text $graphics "combine embeddings" $fontSmall $brushMuted 1098 438
Center-Text $graphics "HLT + reco" $fontMid $brushGold 1098 477

Draw-RoundedRect $graphics 1340 365 180 170 28 $brushPanel $penBlue
Center-Text $graphics "Prediction" $fontLabel $brushInk 1430 400
Center-Text $graphics "jet scores" $fontMid $brushBlue 1430 462

Draw-Arrow $graphics 410 327 575 295 $penMuted
Draw-Arrow $graphics 410 650 575 615 $penMuted
Draw-Arrow $graphics 845 295 965 410 $penMuted
Draw-Arrow $graphics 845 615 965 490 $penMuted
Draw-Arrow $graphics 1230 450 1340 450 $penMuted

$graphics.DrawString("two views of the same jet", $fontSmall, $brushMuted, 140, 486)
$graphics.DrawString("fusion happens after each view has its own representation", $fontSubtitle, $brushInk, 430, 830)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
