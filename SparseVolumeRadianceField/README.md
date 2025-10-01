## Git clone
```bash
git clone https://github.com/nortonBryan/Clear-plenoxels.git
```
## Environment configuration
```bash
pip install -r requirements.txt
```
## Download NeRF Datasets
### NeRF Official link:
```bash
https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4
```
### Transfer image to text formats (for transplantation)
```bash
python utils/preprocess_datasets.py --dataset_dir path/to/your/NeRF Blender Dataset
```
### Extract Camera pose files:
```bash
unzip Poses.zip -d path/to/your/NeRF Blender Dataset
```


## Training
```bash
cd SparseVolumeRadianceField/
make clean
make
```
### Edit the configuration file "SparseVolumeRadianceField/config_linux.txt" in line 18,19,20,21 and 40 to your path accordingly

### Training command:
```bash
./VFR config_linux.txt 0
```
where the last number is your GPU device id you want to use

### Visualize training process and results
```bash
python utils/metrics.py --exp_dir path/to/you/specifiedpathonline_40_in config_linux.txt --dataset_dir path/to/your/NeRFDatasetspath
```
