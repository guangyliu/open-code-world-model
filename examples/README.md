# Released example conditions

This repository does not vendor the authors' data. The 40 released examples -- their proxy
conditions, prompts and first frames -- come from the Code World Model release:

    https://github.com/buaacyw/code-world-model
    https://huggingface.co/datasets/buaacyw/code-world-model-examples

Download them and point the driver at the directory:

    export CWM_EXAMPLES=/path/to/examples40
    python portb/run_cp.py --example 2 --window 0 ...

Expected layout, one directory per example:

    examples40/<N>/metadata.json
    examples40/<N>/first_frame.png
    examples40/<N>/prompts/NNN.txt
    examples40/<N>/condition/NNN.npz     # keys: depth, semantic_id

Two things to know before you parse them, both measured rather than documented upstream:

  * `metadata.json` carries a per-example `depth_format`. 15 of the 40 are metric float32;
    the other 25 store uint16 fixed-log codes in a float32 array and use a different 2x2
    resample. Read the field -- do not infer the family from the array dtype.
  * Examples 32 and 33 ship byte-identical conditions with different first frames, so the
    40 directories contain 39 distinct geometries.
