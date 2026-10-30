# Griding verification notes (2026-10-29)

## What the tool does

Covers people's faces in live-action plates with a grid, a solid mask, or face points and lines, so the footage can go into Seedance.
Runs on Windows from `Griding.bat` alone, without ComfyUI (Python, models and ffmpeg are all inside the folder).

## Models

| Task | Model |
|---|---|
| Head detection | YOLO11s / YOLO11m head (deepghs) · graphics card (DirectML) or CPU |
| Linking people | ByteTrack-style tracker (no model, written for Griding) |
| Face landmarks | Google MediaPipe Face Landmarker (478 points) · CPU |
| Read · write | FFmpeg (H.264 · lossless PNG MOV) |

## 1. Seedance content-filter test

Seedance 2.5 · reference-to-video · 480p · 9-second Paris plate, 23 people

| Test | Result |
|---|---|
| T0 no cover | Rejected (real person) |
| T1 whole clip · thin red grid | Rejected — eyes, nose and mouth visible between grid lines |
| T2 grid on the first frame only | Rejected |
| T3 grid on large faces only | Rejected |
| T4 grid on faces over 100 px only | Rejected |
| T5 whole clip · solid black whole-head mask | Passed — bodies may stay visible; new faces were generated following the reference blocking and camera |

- 2.5 filters on whether a face is visible. For Seedance, use the "Mask · whole head" or "Mask · face outline" card.
- Higgsfield uses a different check path, so a grid might pass there — not verified.
- Files: `Downloads\검열테스트_131019\` (video · prompt · T5 result)

## 2. Source → result spec (measured)

| Item | Result |
|---|---|
| Resolution · fps · frame count | Same as source (217 frames · 1920×1080 · 24 fps; values like 23.976 are exact too) |
| Color | Before the fix: BT.709 video was read as BT.601 and written about 2.5 levels dark. After: brightness bias +0.05, average difference about 1 (H.264 loss only) · color space tags written |
| Audio | AAC, MP3 and AC3 copied as is; PCM and others converted to AAC |
| Rotation · variable fps · odd sizes | Rotation applied · converted to constant fps · mp4 padded 1 px to even (layers keep source size) |
| Check after writing | Each file's frames, size and fps compared to the source; "✓ matches source spec" on screen |
| Bitrate | Constant CBR — 2K 30 Mbps · 4K 50 Mbps (or the source bitrate, held constant). Measured on a 9-second plate: 2K 29.3 Mbps · 4K 48.8 Mbps · source 16.5 kept → 16.0 Mbps |

## 3. Per-card transparent layers (composite in an editor, no H.264 pass)

- `_layer1_xxx.mov` … lossless PNG · transparent background · source size · same frame count
- Source + layers stacked in number order = the combined result (average difference 1.1–1.3, H.264 loss on the combined side)
- One layer is 3.6–4.1 MB for a 9-second 1080p plate

## 4. Heavy files

| Input | Result |
|---|---|
| ProRes 422 HQ MOV 200 MB · 10-bit · PCM audio | All 4 outputs match spec · 79 s · peak memory 500 MB |
| 4K 60 fps HEVC 10-bit MOV, 9 s (540 frames) | All 4 outputs match spec · about 12 min · peak memory 840 MB |

- 4K is much slower than 1080p. For Seedance, scale down to 1080p before adding the plate.
- Speed: drawing only where there is something to draw made it about 2× faster (pixel-identical output).
- MOVs the browser cannot play (ProRes etc.) still process fine; the screen says so.
- Seedance input limits checked: 200 MB · 15 s for 2.0 / 30 s for 2.5 · each side 300–6000 px · aspect 0.4–2.5

## 5. On another PC

- Extract the zip → `Griding.bat` (`Griding_CPU.bat` if the graphics card causes trouble)
- 9-point PC check on every launch (OS · CPU/memory · runtime · components · ffmpeg · models · graphics card · face landmarks · disk space)
- Missing or failing pieces: required ones stop the launch with a fix; optional ones switch off just that feature; a graphics card failure falls back to CPU automatically
- Checked: extracted into a folder with Korean characters and spaces on a clean environment; starts in both GPU and CPU mode; 155 tests pass
- Not done yet: running on an actual second PC

## 6. Automated tests (176)

Spec · bitrate · mask opacity · shortcut · color round trip (709/601) · variable fps · rotation · 10-bit · odd sizes · transparent layers · people tracking · face landmarks · server · PC check · blending speed · sample character cards · result versioning · saving next to the source · file picker · face line/point opacity · tracking box color · macOS code paths with mocks (file dialog, Finder, Core ML, system check, launchers, Mac zip permissions)

Run: `app\python\python.exe app\tests\run_tests.py`

## 7. macOS builds (built on Windows, run on a Mac 2026-10-30)

- `Griding_1.0_mac_arm64.zip` (Apple silicon, 284 MB): Python 3.12.16 (python-build-standalone) and all 21 libraries at the same versions as Windows (onnxruntime instead of onnxruntime-directml)
- `Griding_1.0_mac_intel.zip` (244 MB): onnxruntime 1.23.2 (the last Intel-Mac build) and no MediaPipe, so the face cards are off
- Checked on Windows: every native file is a Mach-O binary for the right CPU; `Griding.command`, `python3.12` and `ffmpeg` carry execute bits; symlinks are stored as symlinks; the launchers pass `bash -n`; all needed modules are present; wheel tags match the target
- Minimum macOS 14, because onnxruntime 1.24 (Apple silicon) and OpenCV 5 (Intel) need it
- Run on a real Mac by the user (2026-10-30): starts and works
