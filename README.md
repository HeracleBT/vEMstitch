# vEMstitch

Here is an official code for vEMstitch, an algorithm for fully automatic image stitching of volume electron microscopy.

## Requirements

We have provided the "environment.yaml" for Anaconda.

```
conda env create -f environment.yaml 
```

## Usage

In the "source" directory, there is a main.py file.

```
Usage: python source/main.py --input_path [data dir] --store_path [result dir] [(options)]

---options---
--pattern [int (N)]: The N*N image stitching
--refine: feature re-extraction or not
```

We strongly recommend that user first use the tool without "--refine" to fast and robustly stitch tiles. Then, if there are misalignments in some results, users can specifically re-run the tool with "--refine" to acquire seamless images.

## Testing example

We also provide a testing example in the "test" directory.

### Without refinement
```
python source/main.py --input_path test --store_path test_res --pattern 3
```

### With refinement
```
python source/main.py --input_path test --store_path test_res --pattern 3 --refine
```

### simulated_data
for illustration, some simulation results

raw_data: raw single image

simulation1: translation only

simulation2: translation + rotation

simulation3: rigid + local distortion

C*_*: section image

C*_res: result of vEMstitch

fiji_restore: result of Fiji

mist_restore: result of MIST

trakem2_restore: result of TrakEM2
