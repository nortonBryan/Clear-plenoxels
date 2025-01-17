## Git clone

```bash
git clone https://github.com/nortonBryan/Clear-plenoxels.git
```
## NeRF Datasets prepare
### Download NeRF Datasets in:
```bash
https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4
```
### Transfer image to text formats (Since we do not use any other libaray like opencv for C/C++)
```bash
python utils/preprocess_datasets.py --dataset_dir path/to/your/NeRF Blender Dataset
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