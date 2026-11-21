VESUVIUS READER - look for ink in the Herculaneum scrolls on an ordinary Windows laptop
=====================================================================================

No graphics card, no Python install, no command line.

1. Unzip this folder somewhere short, for example C:\VesuviusReader or your Desktop.
   (Right-click the zip > Extract All. It has many small files, so this can take a few minutes.)
2. Double-click "Vesuvius Reader.bat".
   - If Windows says "Windows protected your PC", click "More info" and then "Run anyway".
     This app is new and not code-signed yet; the source is public at
     https://github.com/jamiejmann/vesuvius-cpu-reader
   - A black window opens (keep it open) and then your web browser shows the app.
     Everything runs on your own computer; the browser page is just the window.
3. Pick a segment from the list, optionally drag a box on the preview to choose a smaller area,
   and press "Read the ink". The first run downloads the Reader V3 model (about 140 MB).
4. Your images are saved in Documents\Vesuvius Reader. White marks are likely ink.

How long does it take? On an AMD Ryzen 5 7520U laptop, a 2048 x 4096 pixel area took about 11 minutes.
Smaller boxes finish in a minute or two. Close other heavy programs for the best speed.

To quit, close the black window. If something goes wrong, the "last-run-log.txt" file in
Documents\Vesuvius Reader has the details - please include it when you report a problem on
GitHub or in the Vesuvius Challenge Discord.

"Self-test.bat" runs a tiny check (a 384 x 768 area of PHerc1447 w058) and should finish with "ok": true.

Credits: Reader V3 by Domenico Russo; inference code from the Vesuvius Challenge team's villa repository;
scroll data from the Vesuvius Challenge (CC BY-NC 4.0). See the licenses folder.
