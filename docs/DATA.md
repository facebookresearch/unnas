# Setting Up Data Paths

Expected datasets structure for CIFAR-10:

```
cifar10
|_ data_batch_1
|_ data_batch_2
|_ data_batch_3
|_ data_batch_4
|_ data_batch_5
|_ test_batch
|_ ...
```

Expected datasets structure for ImageNet:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Expected datasets structure for ImageNet-22K:

```
imagenet22k
|_ n01440764
|_ ...
|_ n15075141
|_ ...
```

Commands for setting up Cityscapes:

```
cd /path/cityscapes
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
git clone https://github.com/mcordts/cityscapesScripts.git
mv cityscapesScripts/cityscapesscripts ./
rm -rf cityscapesScripts
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

Create a directory containing symlinks:

```
mkdir -p /path/unnas/pycls/datasets/data
```

Symlink CIFAR-10:

```
ln -s /path/cifar10 /path/unnas/pycls/datasets/data/cifar10
```

Symlink ImageNet:

```
ln -s /path/imagenet /path/unnas/pycls/datasets/data/imagenet
```

Symlink ImageNet-22K:

```
cd /path/unnas/pycls/datasets/data
wget https://dl.fbaipublicfiles.com/unnas/imagenet-22k.txt
ln -s /path/imagenet22k /path/unnas/pycls/datasets/data/imagenet22k
```

Symlink Cityscapes:

```
ln -s /path/cityscapes /path/unnas/pycls/datasets/data/cityscapes
```