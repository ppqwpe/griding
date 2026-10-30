# Griding 1.0 — How to use

Griding covers the faces in live-action plates with a grid or a solid mask so the footage can go into Seedance.
Nothing to install. Windows 10/11.

## Start

Double-click `Griding.bat`. A black console window opens, the PC check runs, and the app opens in your browser.
The first launch puts a Griding shortcut on the desktop; use that from then on.
If the graphics card makes it hang or crash, start `Griding_CPU.bat` instead. Same results, a little slower.

Keep the console window open while you work. If the browser does not open, paste the address shown in the console
(something like `http://127.0.0.1:7860/`) into Chrome or Edge.
Switch the language with **[한국어 | English]** at the top right.

## Screen layout

Griding is laid out like an editing workspace (think Premiere). From 1200 px wide it fills the whole window, with docked panels and no page scroll:

| Left | Center | Cards | Export |
|---|---|---|---|
| ② People, and below it ① the plate list | The drop box, the **Monitor** (source \| result), the run bar (⑤ Preview this frame · Run all), and below it ⑤ Run · Results with the **results list** | ③ Cards, and below them the card settings (colors and opacity, head detection, how it works) | **④ Export** |

- Drag the thin lines between panels to resize them: the widths of the left, cards and export columns, the monitor height, and the height of the plate list. Double-click a line to reset it.
- Panel sizes are remembered in this browser.
- On smaller screens the side columns shrink first so the monitor keeps its room.
- Below 1200 px the panels stack in order.

**Results list** — every run adds a row: plate, version, time, combo, where it was saved, number of files, and the spec check (e.g. `2/2`). **View** opens that plate and version in the monitor; **Download** and **Folder** work as in the plate list.

## Steps

### ① Add plates

**Click the box** to open the Windows file dialog, or **drag files** onto it. Up to 20 plates at a time.

- Files you **choose in the dialog** keep their real location, so results can be saved next to them (see ⑤). They are not copied.
- Files you **drag in** are uploaded to the app. The browser does not tell the app where they came from, so their results go to the `results` folder.

Each plate gets a row in the list: thumbnail, name, spec, status (Waiting · Processing % · Done · Error), and buttons for **Download**, **Folder**, **Run again** and **✕** (removes it from the list; the file stays). Click a row to work on that plate.
Below the source you see resolution, fps, length, color space and audio. Anything worth knowing is shown in yellow.

### ② Check people

As soon as a plate is in, the whole video is scanned for people and each one gets a number (#1, #2 …) and a face thumbnail.
Keep the people you want covered and click the rest to drop them. After a cut, the same person gets a new number.

### ③ Pick and stack cards

- Start from a preset button (Grid only, Masked face guide …) or click cards to stack them.
- Cards picked later sit on top. Reorder them with ↑↓ under **[Stack order]**.
- Choose the background: over the source, or black.
- **[Card settings]** shows settings only for the cards you picked. Masks, the grid, and face lines/points each have their own color palette and opacity.
- Before a plate is added, the preview uses the sample character.
- **[Use this combo for: All plates | This plate only]** — by default every plate uses the same combo. Pick **This plate only** to give the selected plate its own combo; it is marked **Own combo** in the plate list, and switching back to **All plates** returns it to the shared one. Each plate runs with its own combo when you press **[Run all]**.

### ④ Export

Tick the files you want. You can pick several.

| File | What it is |
|---|---|
| Combined video | One mp4 with all picked cards stacked. This is the file for Seedance |
| Per-card transparent layers | One transparent MOV per card, for compositing in After Effects or Premiere |
| Check files | Source \| result \| tracking comparison video, still, 9-frame sheet |

Combined video settings

| Setting | What it does |
|---|---|
| Match source | Resolution, fps, color space and audio are locked to the source (lock icon) |
| Bitrate | Constant (CBR). **[By resolution]** 30 Mbps up to 2K, 50 Mbps at 4K / **[Same as source]** the source video's bitrate |
| Estimated size | Estimates the file size and tells you whether it fits Seedance's 200 MB limit |

Open **[Spec details]** to see, item by item, how the source maps to the result.

### ⑤ Run and download

Check one frame with **[Preview this frame]**, then press **[Run all]**.
When it finishes, each file shows a "matches source spec" check and a **[Download]** button.

**Save to**

| Choice | Where the files go |
|---|---|
| Source folder (suffix on the name) | Next to the original: `D:\shots\plate.mp4` → `D:\shots\plate_combo.mp4`, `plate_layer1_grid.mov`, `plate_settings.json` … Files are written to a hidden temporary folder first and moved in only when the run succeeds, so a failed run leaves nothing behind. If the folder is not writable, or the plate was dragged in, the results go to the `results` folder instead and the screen says so |
| results folder | A numbered folder per plate: `results\01_plate\plate_combo.mp4` |

**On a re-run** — **[New version (_v2 · _v3)]** keeps earlier results (`plate_v2_combo.mp4`, or `results\01_plate_v2\`); **[Overwrite]** replaces files with the same names.
The line under the buttons shows the actual file name the selected plate will get.

**Versions** — every run of a plate is listed under ⑤ as `v1`, `v2` … with the time, the combo used and where it was saved. Click a version to see its files again. The plate list shows the latest version number. Use **[Run again]** in the list to redo just one plate.

## Cards

| Card | What it does |
|---|---|
| Mask · face outline | Covers the face outline, slightly widened. Profiles and the back of the head, where no face is found, are covered with the head box |
| Mask · whole head | Covers the whole head including ears and hair. Works without face landmarks |
| Grid | Puts a grid over the head. 8 patterns (grid, diagonal grid, horizontal lines, vertical lines, dots, checker, X cross, filled box) |
| Face lines | Eyes, brows, mouth, nose and face outline (red by default, opacity adjustable) |
| Face points | 478 face points (red by default, opacity adjustable) |
| Tracking boxes | Detected head boxes with numbers, in a color and opacity you choose (green by default). Orange (HOLD) = briefly lost and held |

- Mask opacity below 100% lets the original face show through. If a face is visible, Seedance may reject the clip.
- Face lines and points only appear from front to three-quarter views, where eyes and mouth are visible. Empty frames for full profiles or the back of the head are expected.

## Grid only

In ③, press **[Grid only (Seedance)]** to use the grid card alone.
The combined video is the grid over the source, and the transparent layer is a MOV with only the grid.
For a white grid on black (grid matte), turn on "Grid matte" under ④ › **[More outputs]**.

## Compositing the per-card layers

`name_layer1_cover_face.mov`, `name_layer2_points.mov` … The number is the stacking order.
PNG-codec MOV, RGBA (transparent background), lossless, same size and frame count as the source. No audio.

Put the source plate at the bottom of the timeline and lay layer1, layer2 … on top at the same start (blend mode Normal).
The result matches the combined video. The source stays at its full 10-bit quality, without an extra H.264 pass.
A layer is about 4 MB for a 9-second 1080p plate (measured 3.6–4.1 MB). Browsers cannot preview these; open them from the results folder.

## How the source spec is kept

| Stage | What happens |
|---|---|
| Read | Decodes with the source's color space (BT.709, BT.601, BT.2020) and range (tv, pc). Applies rotation metadata. Variable frame rate (phones, screen recordings) is converted to constant frames at the base fps so audio stays in sync |
| Write | Resolution, fps (including values like 23.976), color space and tags, and pixel aspect (SAR) match the source. AAC, MP3 and AC3 audio is copied without re-encoding; anything else becomes AAC 192k. Odd widths or heights get 1 px added for the mp4 only (an H.264 rule); transparent layers keep the source size. 10-bit sources become 8-bit in the mp4; composite the transparent layers if you need full quality |
| Compare | After writing, each file's frame count, size and fps are compared to the source, shown on screen and saved to `_settings.json` |

Measured (9-second plate)

- 2K fixed → 29.3 Mbps average · 4K fixed → 48.8 Mbps · source 16.5 Mbps kept → 16.0 Mbps
- Color: average difference from the source in unmasked areas about 1 (out of 0–255, H.264 loss), brightness bias +0.05

## Results folder

`results\01_name\` — numbered 01, 02 … in the order plates were added. A re-run goes to `01_name_v2`.

| File | What it is |
|---|---|
| `name_combo.mp4` | Combined video |
| `name_layerN_card.mov` | Per-card transparent layer |
| `name_compare.mp4` | Source \| result \| tracking, 3-up comparison |
| `name_preview.jpg` · `name_contact.jpg` | Comparison still · 9-frame sheet |
| `name_settings.json` | Settings, input spec, output check, head positions per frame |
| `results\verification.json` | Summary of all runs |

**[More outputs]** can also write the older file types:
grid only (`_facegrid`), tracking check (`_overlay`), grid matte (`_matte`), masked face guide (`_masked`), face guide (`_face`),
face tracking check (`_face_overlay`), face points (`_points` + coordinates `_points.json`), face points check (`_points_overlay`),
cover (`_cover`), head mask (`_headmask`).

## Troubleshooting

| Problem | What to do |
|---|---|
| Heads get lost in some shots | In ③ › **[Head detection · padding · smoothing]**, lower detection sensitivity to 0.15 |
| Wrong things get detected | Raise detection sensitivity to 0.35 |
| Small heads, overlapping people | Switch the head model to "Accurate (YOLO11m)". 2–3× slower |
| Hangs on the graphics card | Start with `Griding_CPU.bat`. If the driver stops mid-job, Griding switches to CPU and carries on |
| The screen looks outdated | Close the console window and start again from the Griding desktop shortcut |

The PC is checked on every launch ("PC check" at the bottom of the screen, `work\pc_check.json`).
✓ works, ! works with a note, ✗ unavailable — unavailable items come with what to do.

| Situation | What the app does |
|---|---|
| Fast head model, ffmpeg or components missing | Does not start; tells you to copy the whole folder again |
| Accurate model missing | Uses "Fast" only |
| Face landmarks fail to load | Only the face cards are unavailable. Grid and whole-head mask still work |
| App folder is not writable | Results move to `Documents\Griding results` automatically |
| Run straight from inside the zip | Tells you to extract first |

If it still fails, run the self-test: `app\python\python.exe app\tests\run_tests.py` (164 tests)

## Moving to another PC

Copy the whole folder, or **Extract All** from the release zip, then run `Griding.bat`. About 580 MB.
No ComfyUI, Python, Visual C++ install or graphics card needed.
Any graphics card is used through DirectML; without one, the CPU is 1.5–3× slower.
Folders with Korean characters or spaces in the path are fine.

Download: <https://github.com/ppqwpe/griding/releases>

## Models

| Task | Model |
|---|---|
| Head detection | YOLO11s head (Fast, default) / YOLO11m head (Accurate) — deepghs/real_head_detection, ONNX Runtime |
| Linking people across frames | ByteTrack-style tracker written for Griding (no model). A person must persist for 3+ frames to count |
| Face landmarks | Google MediaPipe Face Landmarker, 478 points, CPU (only when a face card is picked) |
| Video read · write | FFmpeg 7.1 |

Licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The head models' authors do not state a license. They were trained with Ultralytics (AGPL-3.0), so treat them as AGPL-3.0.
FFmpeg is GPL-3.0.
