param(
    [string]$Output = "teacher_logit_reco/presentation_assets/model_architecture_simple.jpg"
)

Add-Type -AssemblyName System.Drawing

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$outPath = Join-Path $root $Output
$outDir = Split-Path -Parent $outPath
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$width = 900
$height = 1360
$bitmap = New-Object System.Drawing.Bitmap $width, $height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

$bg = [System.Drawing.Color]::FromArgb(248, 245, 238)
$ink = [System.Drawing.Color]::FromArgb(24, 35, 46)
$muted = [System.Drawing.Color]::FromArgb(84, 99, 115)
$orange = [System.Drawing.Color]::FromArgb(228, 87, 46)
$orangeSoft = [System.Drawing.Color]::FromArgb(250, 224, 213)
$gold = [System.Drawing.Color]::FromArgb(202, 146, 54)
$goldSoft = [System.Drawing.Color]::FromArgb(254, 244, 216)
$green = [System.Drawing.Color]::FromArgb(36, 148, 99)
$greenSoft = [System.Drawing.Color]::FromArgb(220, 242, 231)
$teal = [System.Drawing.Color]::FromArgb(20, 145, 151)
$tealSoft = [System.Drawing.Color]::FromArgb(216, 242, 243)
$purple = [System.Drawing.Color]::FromArgb(116, 91, 184)
$purpleSoft = [System.Drawing.Color]::FromArgb(234, 229, 248)
$panel = [System.Drawing.Color]::FromArgb(255, 255, 255)

$graphics.Clear($bg)

$fontTitle = New-Object System.Drawing.Font("Arial", 31, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Regular)
$fontLabel = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Bold)
$fontMid = New-Object System.Drawing.Font("Arial", 17, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Regular)

$brushInk = New-Object System.Drawing.SolidBrush($ink)
$brushMuted = New-Object System.Drawing.SolidBrush($muted)
$brushOrange = New-Object System.Drawing.SolidBrush($orange)
$brushOrangeSoft = New-Object System.Drawing.SolidBrush($orangeSoft)
$brushGoldSoft = New-Object System.Drawing.SolidBrush($goldSoft)
$brushGreen = New-Object System.Drawing.SolidBrush($green)
$brushGreenSoft = New-Object System.Drawing.SolidBrush($greenSoft)
$brushTeal = New-Object System.Drawing.SolidBrush($teal)
$brushTealSoft = New-Object System.Drawing.SolidBrush($tealSoft)
$brushPurple = New-Object System.Drawing.SolidBrush($purple)
$brushPurpleSoft = New-Object System.Drawing.SolidBrush($purpleSoft)
$brushPanel = New-Object System.Drawing.SolidBrush($panel)

$penInk = New-Object System.Drawing.Pen($ink, 2.0)
$penMuted = New-Object System.Drawing.Pen($muted, 2.5)
$penOrange = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(236, 135, 95), 2.0)
$penGold = New-Object System.Drawing.Pen($gold, 2.2)
$penGreen = New-Object System.Drawing.Pen($green, 2.2)
$penTeal = New-Object System.Drawing.Pen($teal, 2.2)
$penPurple = New-Object System.Drawing.Pen($purple, 2.2)

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

function Draw-JetMini {
    param($G, [int]$Seed, [float]$Cx, [float]$Cy, $ParticleBrush, $FillBrush, $Pen, [int]$N)
    $rng = [System.Random]::new($Seed)
    $G.FillEllipse($FillBrush, $Cx - 88, $Cy - 54, 176, 108)
    $G.DrawEllipse($Pen, $Cx - 88, $Cy - 54, 176, 108)
    for ($i = 0; $i -lt $N; $i++) {
        $r = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $x = $Cx + [Math]::Cos($theta) * $r * (12 + $rng.NextDouble() * 68)
        $y = $Cy + [Math]::Sin($theta) * $r * (9 + $rng.NextDouble() * 40)
        $pt = 3.0 + $rng.NextDouble() * 10.0
        $rr = [Math]::Max(3.0, [Math]::Min(8.5, $pt * 0.68))
        Draw-Particle $G $x $y $rr $ParticleBrush $penInk
    }
}

function Draw-Box {
    param($G, [float]$X, [float]$Y, [float]$W, [float]$H, [string]$Title, [string]$Sub, $Brush, $Pen)
    Draw-RoundedRect $G $X $Y $W $H 22 $Brush $Pen
    Center-Text $G $Title $fontMid $brushInk ($X + $W / 2) ($Y + 24)
    if ($Sub -ne "") {
        Center-Text $G $Sub $fontSmall $brushMuted ($X + $W / 2) ($Y + 56)
    }
}

function Draw-BranchBox {
    param($G, [float]$X, [float]$Y, [string]$Title, [string]$Sub, $Brush, $Pen)
    Draw-RoundedRect $G $X $Y 420 170 26 $Brush $Pen
    Center-Text $G $Title $fontLabel $brushInk ($X + 210) ($Y + 42)
    Center-Text $G $Sub $fontSmall $brushMuted ($X + 210) ($Y + 91)
}

$graphics.DrawString("Reconstructor architecture", $fontTitle, $brushInk, 45, 48)
$graphics.DrawString("Pseudo-HLT particles are encoded once.", $fontSubtitle, $brushMuted, 48, 101)
$graphics.DrawString("Four specialized heads propose candidates and weights.", $fontSubtitle, $brushMuted, 48, 132)

Draw-RoundedRect $graphics 55 245 300 260 28 $brushPanel $penOrange
Center-Text $graphics "Pseudo-HLT" $fontLabel $brushInk 205 282
Center-Text $graphics "particles" $fontSmall $brushMuted 205 318
Draw-JetMini $graphics 13 205 420 $brushOrange $brushOrangeSoft $penOrange 27

Draw-Box $graphics 55 650 310 190 "Shared encoder" "particle + jet context" $brushGoldSoft $penGold

Draw-BranchBox $graphics 430 220 "Edit branch" "adjust existing particles" $brushGreenSoft $penGreen
Draw-BranchBox $graphics 430 510 "Split branch" "parent to children" $brushTealSoft $penTeal
Draw-BranchBox $graphics 430 800 "Generate branch" "new candidates" $brushPurpleSoft $penPurple
Draw-BranchBox $graphics 430 1090 "Budget head" "counts + weights" $brushGoldSoft $penGold

Draw-Arrow $graphics 205 505 205 650 $penMuted

Draw-Arrow $graphics 365 700 430 305 $penMuted
Draw-Arrow $graphics 365 745 430 595 $penMuted
Draw-Arrow $graphics 365 790 430 885 $penMuted
Draw-Arrow $graphics 365 825 430 1175 $penMuted

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
