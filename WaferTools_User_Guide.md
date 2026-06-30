# WaferTools — Quick Start Guide

*No coding required. Everything is done by double-clicking files and clicking buttons in your web browser.*

The program lives in the folder **`D:\WaferTools-main`**.

> **Requirement (one time):** Git must be installed. If it isn't, install it once
> from <https://git-scm.com/download/win> (just click *Next* through the installer).

---

## Step 0 — First-time setup (do this only once per computer)

Pick the **one** option that matches your situation. You only do this once;
afterwards you go straight to Step 1 every time.

### Option A — You already have WaferTools, you just want updates (Sync)

Use this if the folder **`D:\WaferTools-main`** already exists on the computer
(the usual case — e.g. it was installed earlier from a ZIP).

1. Open the folder **`D:\WaferTools-main`**.
2. Double-click **`Setup_Git_Once.bat`**.
3. Wait until it says **"Setup complete"**, then close the window.
   - Your existing data (results, uploads, settings) is **not** touched.

### Option B — Computer doesn't have WaferTools yet (Direct install)

1. Ask Fuming for **`Install.bat`** (one small file). Save it where you want
   WaferTools to live, e.g. directly on **`D:\`**.
2. Double-click **`Install.bat`**. It downloads the program into a new
   **`WaferTools`** folder next to it.
3. Done — that folder is already update-ready, so you can skip to Step 2 the
   first time (no separate setup needed).

> Either way, the **first time** the program is fetched onto the computer once.
> After Step 0 you never download anything by hand again — updates are one click.
>
> *Note:* depending on how it was installed, your folder may be named
> **`WaferTools-main`** or **`WaferTools`**. The files inside (`Update.bat`,
> `WaferTools.bat`, …) are identical.

---

## Step 1 — Update to the latest version (Sync)

Do this whenever Fuming tells you there is a new version (after Step 0 is done once).

1. Open the folder **`D:\WaferTools-main`**.
2. Double-click **`Update.bat`**.
3. A black window appears and downloads the latest version. When it says
   **"You now have the latest version"**, you can close the window.

> If it says it can't update, make sure you did **Step 0** once. Your current
> version keeps working either way — if unsure, contact Fuming.

---

## Step 2 — Start the program (Launch)

1. In the same folder, double-click **`WaferTools.bat`**.
2. A black window opens.
   - **The very first launch takes a few minutes** (it sets things up). Later launches are quick.
3. Your web browser opens automatically at **`http://127.0.0.1:8050`**.
   - If it doesn't open by itself, open your browser and type that address.
4. **Keep the black window open while you work.** Closing it stops the program.

When you are finished, just close the black window.

---

## Step 3 — One-time setup: where results are saved (NEW)

On the **left sidebar** you'll see two new boxes:

1. **Results folder** — type the folder where you want all outputs saved,
   for example `D:\WaferData\Results`. Leave it blank to use the default.
2. **Project / wafer name** — *leave this blank.* The program will name
   everything automatically from the file you load (see Step 4). Only type a
   name here if you want to force a specific label for every export.
3. Click **Save settings**. Your choice is remembered the next time you launch.

The small line **"Current: …"** below the button shows the wafer name the
program is using right now.

---

## Step 4 — How your files are named now (NEW)

You no longer get folders that are just dates. Every run is now labeled with the
**wafer name**, which the program reads automatically from the image or folder
you load.

**Example** — if you load an image called `wafer 14.png`, your results are saved like this:

```
D:\WaferData\Results\
  section_counter\
    wafer 14_20260726_0930\      <-- folder = wafer name + date/time
      wafer 14_sections.csv      <-- files start with the wafer name
      wafer 14_mask.png
```

The same naming is used by all four tools (Section Segmentation, ROI
Registration, Section Ordering, Order Visualization).

---

## Step 5 — The four tools

Use the buttons in the **left sidebar** to switch between tools.

### A. Section Segmentation
1. **Upload Image** — choose your wafer image.
2. **Auto-loading Cache Detection** (or **Run New Detection** for a fresh run).
3. Adjust the **Filtering** slider if needed.
4. **Export** — saves into `…\section_counter\<wafer>_<date>\`.

### B. ROI Registration
1. **Upload Image** and **Upload CSV**.
2. **Unify All Masks**.
3. **Confirm & Export** — saves into `…\mask_unification\<wafer>_<date>\`.

### C. Section Ordering  ⭐ (now fully automatic)
1. Type or paste the **Images Folder Path** and click **Scan Images**.
2. (Optional) adjust the SIFT sliders — the defaults are fine for most wafers.
3. Click **Run SIFT Pairwise Alignment**.
4. **That's it — just wait.** When alignment finishes, the program now
   **automatically cleans the data and builds the section order for you.**
   The final order appears in the results box at the bottom.
   - You **no longer** have to pick a CSV file or choose a date folder.
   - The *Upload raw CSV / Clean / Build* buttons are still there only as a
     manual backup; you normally won't need them.
5. Results are saved into `…\sequencing\`.

### D. Order Visualization
1. **Upload Images**, **Upload Mask CSVs**, **Upload Chain TXT**.
2. **Generate Overlay** or **Generate Stack Alignment** to view the result.
3. Output is saved into `…\order_viz\`.

---

## Step 6 — Finding your results

Open your **Results folder** → open the tool's subfolder → open the
`<wafer>_<date>` folder. Everything inside is prefixed with the wafer name.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| Browser didn't open | Open it and type `http://127.0.0.1:8050` |
| "Port already in use" message | It's already running — just open the address above |
| Black window closed by accident | Double-click `WaferTools.bat` again |
| `Update.bat` can't update | Your current version still works; contact Fuming |
| Something looks wrong | Take a photo of the black window and send it to Fuming |

---

*Questions or new requests? Send them to Fuming.*
