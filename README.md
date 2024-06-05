# vEMstitch

Here is an official code for vEMstitch, an algorithm for fully automatic image stitching of volume electron microscopy. We provide both python versio and C++ version.

## Requirements

Python:

We have provided the "environment.yaml" for Anaconda.

```
conda env create -f environment.yaml 
```

C++:

OpenCV, cmake

## Usage

Python:

In the "source" directory, there is a main.py file.

```
Usage: python source/main.py --input_path [data dir] --store_path [result dir] [(options)]

---options---
--pattern [int (N)]: The N*N image stitching
--refine: feature re-extraction or not
```

C++:

```
cd vEMstitch_c++
mkdir build
cd build
cmake ..
make

cd ../../
vEMstitch_c++/bin/vEMstitching [data_dir] [store_dir] [pattern] [overlapping_rate] [log_dir] [refine_flag]
```

We strongly recommend that user first use the tool without "--refine" to fast and robustly stitch tiles. Then, if there are misalignments in some results, users can specifically re-run the tool with "--refine" to acquire seamless images.

## Testing example

We also provide a testing example in the "test" directory.

### Without refinement

Python:
```
python source/main.py --input_path test --store_path test_res --pattern 3
```
C++:
```
vEMstitch_c++/vEMstitching test test_res 3 0.1 log.txt false
```

### With refinement
Python:
```
python source/main.py --input_path test --store_path test_res --pattern 3 --refine
```
C++:
```
vEMstitch_c++/vEMstitching test test_res 3 0.1 log.txt true
```

## related_data

### simulated_data
for illustration, some simulation results (three examples of total 100 ones)

raw_data: raw single image

simulation1: translation only

simulation2: translation + rotation

simulation3: rigid + local distortion

simulation_noise: different level of noise

simulation_deformation: different level of deformation

C*_*: section image

C*_res: result of vEMstitch

C*_stitching_row*: row result of vEMstitch

fiji_restore: result of Fiji

mist_restore: result of MIST

trakem2_restore: result of TrakEM2

### real_test
we have provided the raw sections and stitched results of compared methods.

The all real mussel images used in the paper are available at https://pan.quark.cn/s/f097018cdf7b.


## Licenses
[![CC-BY-SA](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-sa/4.0/)
[![GPL-3.0](https://img.shields.io/badge/license-GPL-blue.svg)]

<!-- The data is licensed under [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0). -->

We assign Licenses to the code and data separately.
The code matched by the following patterns are licensed under GPL-3.0:

+ `*.py`
+ `*.yaml`
+ `*.cpp`
+ `*.h`
+ `*CMakeLists.txt`

The simulation data based on CREMI dataset(https://cremi.org/data) and real mussel images are available under CC BY 4.0, including:

+ `*.bmp`