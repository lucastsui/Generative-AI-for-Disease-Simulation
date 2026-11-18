# Joint Segmentation

Segments the two bones of a finger joint on hand X-ray crops: the bone above the joint and the
bone below it.

## Try it in the browser

Open `segment.html`. Drop an X-ray crop on the left box, or press "or upload". The right box shows
the same crop with the upper bone in blue and the lower bone in orange. Click the result to switch
between filled bones and outlines.

The page runs the model inside the browser. It needs no server and no network, and the image
stays on your computer. Keep `segment.html` next to the `demo/` folder.

Adding `?selftest` to the address runs the samples and counts the pixels that differ from the
label maps PyTorch gave for them.

## Run it in Python

```sh
pip install -r requirements.txt
python segment.py path/to/joint.png --out results
```

For every image this writes `<name>_mask.png` (180 x 180 label map: 0 background, 1 upper bone,
2 lower bone) and `<name>_segmentation.png` (the X-ray with both bones coloured). Several images
can be given at once. Tested with Python 3.12 on the CPU.

## What the model expects

A 180 x 180 pixel gray crop of one finger joint, with the joint near the middle and the fingertip
towards the top, at about 0.15 mm per pixel. Other images are cut to their middle square and
resized to 180 x 180, and the results on them are unchecked.

## Files

| Path | What it holds |
|---|---|
| `segment.html` | The page |
| `demo/engine.js` | The model as plain JavaScript, with one small WebAssembly routine for speed |
| `demo/model/f0.js` to `f4.js` | The weights for the page (16-bit floats) |
| `demo/examples.js` | The samples shown on the page |
| `segment.py` | Command-line segmentation |
| `hoaseg/` | The U-Net and the inference steps in PyTorch |
| `checkpoints/fold0.pt` to `fold4.pt` | The weights for PyTorch |

## The model

- Five U-Nets (7.8 million weights each), one per cross-validation fold. Their outputs are
  averaged, each also over a left-right flip of the image. From the result, the largest connected
  part of each bone is kept and its holes are filled.
- Trained on 19,597 crops of DIP, PIP and MCP joints from hand X-rays of the Osteoarthritis
  Initiative: 381 with masks that were reviewed and corrected, and 19,216 with automatic drafts
  (SAM 2.1 followed by a rule that places the joint margins) used as weak labels.
- On 159 joints of participants held out from training, the mean Dice is 0.985 and the lowest is
  0.787. The reference masks for these joints are automatic drafts that were reviewed and
  corrected; in the severe DIP joints a person placed the joint line.
- The page and the Python code compute the same model. On those 159 joints their label maps
  differ by at most 2 pixels.
- Known weakness: a scratch line or the edge of an overlay that crosses a bone can cut its mask
  short. The last sample on the page shows it: it is the held-out joint on which the model does
  worst, and its lower bone is only partly marked.

## Training data

The training crops and masks are not part of this repository. Locally they live in
`training_dataset/` (`images/`, `masks/`, `index.csv`), which git ignores.
