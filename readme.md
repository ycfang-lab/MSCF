# Pose-Transfer-MSCF

* pytorch >=1.0.1
* torchvision
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate



### Data Preperation

#### Market1501
- Download the Market-1501 dataset from [here](http://www.liangzheng.com.cn/Project/project_reid.html). Rename **bounding_box_train** and **bounding_box_test** to **train** and **test**, and put them under the ```market_data``` directory.
- Download train/test splits and train/test key points annotations from [Google Drive](https://drive.google.com/open?id=1YMsYXc41dR3k8YroXeWGh9zweNUQmZBw) or [Baidu Disk](https://pan.baidu.com/s/1fcMwXTUk9XKPLpaJSodTrg), including **market-pairs-train.csv**, **market-pairs-test.csv**, **market-annotation-train.csv**, **market-annotation-train.csv**. Put these four files under the ```market_data``` directory.

#### DeepFashion

- Download the DeepFashion dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). 
- Split the raw images into the train split (`fashion_data/train`) and the test split (`fashion_data/test`). 

### Train Market1501 

```
python train.py --dataroot ../datasets/market_data/ --name market-mshr
```

### Train DeepFashion

```
python train.py --dataroot ../datasets/fashion_data/ --name fashion-mshr
```
### Test Market1501 

```
python test.py --dataroot ../datasets/market_data/ --name market-mshr
```
### Test DeepFashion

```
python test.py --dataroot ../datasets/fashion_data/ --name fashion-mshr
```
### Evaluation
采用 SSIM, mask-SSIM, IS, mask-IS, and PCKh在数据集上进行验证 .
