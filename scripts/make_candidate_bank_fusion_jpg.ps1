param(
    [string]$Output = "teacher_logit_reco/presentation_assets/candidate_bank_fusion_simple.jpg"
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

$fontTitle = New-Object System.Drawing.Font("Arial", 38, [System.Drawing.FontStyle]::Bold)
$fontSubtitle = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Regular)
$fontLabel = New-Object System.Drawing.Font("Arial", 19, [System.Drawing.FontStyle]::Bold)
$fontMid = New-Object System.Drawing.Font("Arial", 16, [System.Drawing.FontStyle]::Bold)
$fontSmall = New-Object System.Drawing.Font("Arial", 13, [System.Drawing.FontStyle]::Regular)

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
$penOrange = New-Object System.Drawing.Pen($orange, 2.0)
$penGold = New-Object System.Drawing.Pen($gold, 2.3)
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
    param($G, [float]$X, [float]$Y, [float]$R, $Brush)
    $G.FillEllipse($Brush, $X - $R, $Y - $R, $R * 2, $R * 2)
    $G.DrawEllipse($penInk, $X - $R, $Y - $R, $R * 2, $R * 2)
}

function Draw-ProposalBox {
    param(
        $G,
        [float]$X,
        [float]$Y,
        [string]$Title,
        [string]$Sub,
        $Fill,
        $Pen,
        $ParticleBrush,
        [int]$Seed
    )
    Draw-RoundedRect $G $X $Y 315 135 22 $Fill $Pen
    $G.DrawString($Title, $fontMid, $brushInk, $X + 22, $Y + 18)
    $G.DrawString($Sub, $fontSmall, $brushMuted, $X + 22, $Y + 49)

    $rng = [System.Random]::new($Seed)
    for ($i = 0; $i -lt 11; $i++) {
        $px = $X + 40 + $i * 23 + ($rng.NextDouble() - 0.5) * 7
        $py = $Y + 96 + ($rng.NextDouble() - 0.5) * 25
        $rr = 4.2 + $rng.NextDouble() * 5.2
        Draw-Particle $G $px $py $rr $ParticleBrush
    }
}

function Draw-MixedCloud {
    param(
        $G,
        [int]$Seed,
        [float]$Cx,
        [float]$Cy,
        [float]$Rx,
        [float]$Ry,
        [int]$Count,
        [bool]$DrawEnvelope
    )
    $rng = [System.Random]::new($Seed)
    if ($DrawEnvelope) {
        $G.FillEllipse($brushOrangeSoft, $Cx - $Rx, $Cy - $Ry, $Rx * 2, $Ry * 2)
        $G.DrawEllipse($penOrange, $Cx - $Rx, $Cy - $Ry, $Rx * 2, $Ry * 2)
    }

    for ($i = 0; $i -lt $Count; $i++) {
        $theta = $rng.NextDouble() * 2.0 * [Math]::PI
        $radius = 1.0 - [Math]::Pow($rng.NextDouble(), 2.0)
        $px = $Cx + [Math]::Cos($theta) * $radius * $Rx * 0.85
        $py = $Cy + [Math]::Sin($theta) * $radius * $Ry * 0.78
        $rr = 3.5 + $rng.NextDouble() * 6.5

        switch (($i + $Seed) % 3) {
            0 { $particleBrush = $brushGreen }
            1 { $particleBrush = $brushTeal }
            default { $particleBrush = $brushPurple }
        }
        Draw-Particle $G $px $py $rr $particleBrush
    }
}

$graphics.DrawString("Candidate bank fusion", $fontTitle, $brushInk, 72, 58)
$graphics.DrawString("branch proposals mix once, then the shared bank is sampled three different ways", $fontSubtitle, $brushMuted, 75, 111)

Draw-ProposalBox $graphics 70 190 "Edit candidates" "adjusted particles" $brushGreenSoft $penGreen $brushGreen 11
Draw-ProposalBox $graphics 70 365 "Split candidates" "child proposals" $brushTealSoft $penTeal $brushTeal 23
Draw-ProposalBox $graphics 70 540 "Generated candidates" "new proposals" $brushPurpleSoft $penPurple $brushPurple 37

Draw-RoundedRect $graphics 115 710 225 72 18 $brushGoldSoft $penGold
Center-Text $graphics "weights + counts" $fontMid $brushInk 227 730

Draw-RoundedRect $graphics 560 185 420 565 28 $brushPanel $penGold
Center-Text $graphics "Candidate bank" $fontLabel $brushInk 770 220
Center-Text $graphics "all proposals together" $fontSmall $brushMuted 770 256
Draw-MixedCloud $graphics 51 770 465 165 150 46 $true
Center-Text $graphics "one mixed pool" $fontMid $brushInk 770 665

Draw-Arrow $graphics 385 257 560 345 $penMuted
Draw-Arrow $graphics 385 432 560 445 $penMuted
Draw-Arrow $graphics 385 607 560 545 $penMuted
Draw-Arrow $graphics 340 746 560 660 $penMuted

Draw-Arrow $graphics 980 350 1130 255 $penMuted
Draw-Arrow $graphics 980 465 1130 465 $penMuted
Draw-Arrow $graphics 980 580 1130 675 $penMuted

Draw-RoundedRect $graphics 1130 170 390 180 24 $brushGreenSoft $penGreen
Center-Text $graphics "View 1" $fontMid $brushInk 1325 190
Draw-MixedCloud $graphics 71 1325 275 150 55 18 $true

Draw-RoundedRect $graphics 1130 375 390 180 24 $brushTealSoft $penTeal
Center-Text $graphics "View 2" $fontMid $brushInk 1325 395
Draw-MixedCloud $graphics 83 1325 480 150 55 23 $true

Draw-RoundedRect $graphics 1130 580 390 180 24 $brushPurpleSoft $penPurple
Center-Text $graphics "View 3" $fontMid $brushInk 1325 600
Draw-MixedCloud $graphics 97 1325 685 150 55 20 $true

Center-Text $graphics "same candidates, different samples" $fontSubtitle $brushInk 1035 805

$encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq "image/jpeg" }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 94L)
$bitmap.Save($outPath, $encoder, $params)

$graphics.Dispose()
$bitmap.Dispose()

Write-Output $outPath
