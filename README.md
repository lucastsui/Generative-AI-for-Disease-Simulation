# Joint Segmentation
## Try it in the browser

Open `segment.html`. Drop an X-ray crop on the left box, or press "or upload". The right box shows
the same crop with the upper bone in blue and the lower bone in orange. Click the result to switch
between filled bones and outlines.

## Run it in Python

```sh
pip install -r requirements.txt
python segment.py path/to/joint.png --out results
```
