param(
    [string]$Output = "teacher_logit_reco/presentation_assets/five_models_fusion.jpg"
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
$fontLabel = New-Object System.Drawing.Font("Arial", 18, [System.Drawing.FontStyle]::Bold)
$fontMid = New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Arial", 13, [System.Drawing.FontStyle]::Regular)
$fontLogits = New-Object System.Drawing.Font("Consolas", 15, [System.Drawing.FontStyle]::Bold)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushBlue = New-Object System.Drawing.SolidBrush($blue)
$brushBlueSoft = New-Object System.Drawing.SolidBrush($blueSoft)
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
$penMuted = New-Object System.Drawing.Pen($muted, 2.4)
$penBlue = New-Object System.Drawing.Pen($blue, 2.0)
$penOrange = New-Object System.Drawing.Pen($orange, 2.0)
$penGreen = New-Object System.Drawing.Pen($green, 2.0)
$penTeal = New-Object System.Drawing.Pen($teal, 2.0)
$penPurple = New-Object System.Drawing.Pen($purple, 2.0)
$penGold = New-Object System.Drawing.Pen($gold, 2.2)

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

function Draw-ModelBox {
    param($G, [float]$X, [float]$Y, [string]$Name, [string]$Sub, $Fill, $Pen, $Brush)
    Draw-RoundedRect $G $X $Y 260 82 18 $Fill $Pen
    Center-Text $G $Name $fontMid $brushInk ($X + 130) ($Y + 16)
    Center-Text $G $Sub $fontSmall $brushMuted ($X + 130) ($Y + 45)
    Draw-RoundedRect $G ($X + 280) ($Y + 15) 96 52 12 $brushPanel $Pen
    Center-Text $G "logits" $fontLogits $Brush ($X + 328) ($Y + 31)
}

$graphics.DrawString("Five models and Fusion", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("the first five reco7 variants feed a lightweight logit fusion", $fontSubtitle, $brushMuted, 75, 111)

$models = @(
    @{Name="m2_antioverlap"; Sub="anti-overlap variant"; Fill=$brushBlueSoft; Pen=$penBlue; Brush=$brushBlue},
    @{Name="m2_base"; Sub="default reco"; Fill=$brushGoldSoft; Pen=$penGold; Brush=$brushGold},
    @{Name="m2_budgetlite"; Sub="relaxed budget"; Fill=$brushGreenSoft; Pen=$penGreen; Brush=$brushGreen},
    @{Name="m2_topk60ish"; Sub="top-k candidate cap"; Fill=$brushTealSoft; Pen=$penTeal; Brush=$brushTeal},
    @{Name="m2_genlow"; Sub="less generation"; Fill=$brushPurpleSoft; Pen=$penPurple; Brush=$brushPurple}
)

$startY = 185
for ($i = 0; $i -lt $models.Count; $i++) {
    $m = $models[$i]
    $y = $startY + $i * 112
    Draw-ModelBox $graphics 105 $y $m.Name $m.Sub $m.Fill $m.Pen $m.Brush
    Draw-Arrow $graphics 485 ($y + 41) 710 450 $penMuted
}

Draw-RoundedRect $graphics 710 245 270 410 28 $brushPanel $penGold
Center-Text $graphics "Prediction cache" $fontLabel $brushInk 845 280
Center-Text $graphics "stack_train" $fontSmall $brushMuted 845 325
Center-Text $graphics "fit fusion weights" $fontSmall $brushMuted 845 350
Center-Text $graphics "stack_val" $fontSmall $brushMuted 845 410
Center-Text $graphics "select fusion" $fontSmall $brushMuted 845 435
Center-Text $graphics "final_test" $fontSmall $brushMuted 845 495
Center-Text $graphics "report once" $fontSmall $brushMuted 845 520

for ($i = 0; $i -lt 5; $i++) {
    $yy = 565 + $i * 16
    $graphics.DrawString("[z1 z2 ... z10]", $fontLogits, $brushMuted, 765, $yy)
}

Draw-Arrow $graphics 985 450 1115 450 $penMuted

Draw-RoundedRect $graphics 1115 330 270 240 28 $brushGoldSoft $penGold
Center-Text $graphics "Fusion model" $fontLabel $brushInk 1250 370
Center-Text $graphics "learned logit combiner" $fontSmall $brushMuted 1250 405
Center-Text $graphics "uses only cached" $fontSmall $brushMuted 1250 458
Center-Text $graphics "model predictions" $fontSmall $brushMuted 1250 483

Draw-Arrow $graphics 1385 450 1490 450 $penMuted
Draw-RoundedRect $graphics 1425 365 120 170 20 $brushPanel $penOrange
Center-Text $graphics "final" $fontLabel $brushInk 1485 407
Center-Text $graphics "class" $fontLabel $brushInk 1485 435
Center-Text $graphics "prediction" $fontSmall $brushMuted 1485 475

$graphics.DrawString("Fusion asks whether reco7 variants make complementary mistakes.", $fontSubtitle, $brushInk, 460, 822)

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
