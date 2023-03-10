# 4D Panoptic Lidar Segmentation for SemanticPOSS

<img width="720" alt="4dmain" src="https://user-images.githubusercontent.com/5329637/124156220-077a2500-daa0-11eb-8d59-6dd5c1455500.png">

<a href="https://mehmetaygun.github.io/4DPLS.html">Project Website with Demo Video</a>.

This repo contains code for 4D Panoptic Lidar Segmentation for SemanticPOSS.  
This repo is modified from <a href="https://github.com/MehmetAygun/4D-PLS">4D-PLS</a>.  
The code is based on the Pytoch implementation of  <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KPConv</a>.

### Installation

```bash
git clone https://github.com/Theia-4869/4D-PLS-POSS.git
cd 4D-PLS-POSS
pip install -r requirements.txt
cd cpp_wrappers
sh compile_wrappers.sh
```

### Data
Download the SemanticPoss to the directory `data/SemanticPoss` with labels from <a href="http://www.poss.pku.edu.cn/OpenDataResource/SemanticPOSS/SemanticPOSS_dataset.zip">here</a>

We have already provided the config `semantic-poss.yaml` in the `SemanticPoss` folder

Then create additional labels using `utils/create_center_label.py`:

```bash
python create_center_label.py
```

The data folder structure should be as follows:

```bash
data/SemanticPoss/
└── semantic-poss.yaml
└── sequences/
    └── 00/
        └── poses.txt
        └── calib.txt
        └── times.txt
        └── velodyne
            ├── 000000.bin
            ...
        └── labels
            ├── 000000.label
            ...
        └── tag
            ├── 000000.tag
            ...
    └── 01/
    ...
    └── 05/
```

### Models

For saving models or using pretrained models create a folder named `results` in main directory. 
You can download a pre-trained model from <a href="https://disk.pku.edu.cn:443/link/17C269A408255A4763387B15B1849394">here</a>.

### Training

For training, you should modify the config parameters in `train_SemanticPoss.py`.
The most important thing that, to get a good performance train the model using `config.pre_train = True` firstly at least for 400 epochs, then train the model using `config.pre_train = False`. 

```bash
python train_SemanticPoss.py
```

This code will generate config file and save the pre-trained models in the results directory.

### Testing

For testing, set the model directory the choosen_log in `test_SemanticPoss.py`, and modify the config parameters as you wish. Then run :

```bash
python test_SemanticPoss.py
```

This will generate semantic and instance predictions for small 4D volumes under the test/model_dir. 
To generate long tracks using small 4D volumes use `stitch_tracklets.py`

```bash
python stitch_tracklets.py --predictions test/model_dir --n_test_frames 4
```
This code will generate predictions in the format of SemanticPoss under test/model_dir/stitch .

### Evaluation

For getting the metrics introduced in the paper, use utils/evaluate_4dpanoptic.py

```bash
python evaluate_4dpanoptic.py --dataset=SemanticPoss_dir --predictions=output_of_stitch_tracket_dir
```
### Citing
If you find the code useful in your research, please consider citing:

	@InProceedings{aygun20214d,
	    author    = {Aygun, Mehmet and Osep, Aljosa and Weber, Mark and Maximov, Maxim and Stachniss, Cyrill and Behley, Jens and Leal-Taixe, Laura},
	    title     = {4D Panoptic LiDAR Segmentation},
	    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	    month     = {June},
	    year      = {2021},
	    pages     = {5527-5537}
	}
	
#### License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2021 Mehmet Aygun
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
