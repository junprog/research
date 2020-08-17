# Density Estimation for Crowd Counting

群衆カウントを回帰タスクを用いたFCN(Fully Convolution Net)で実装

## Transform method

* Croping method
    * Random_Crop
    * Corner_Center_Crop

* Gaussian Filtering method
    * Gaussian_Filtering
    
        using under this library
        ```python
        from scipy.ndimage.filters import gaussian_filter
        ```

* Scaling method
    * Scale (not recommended)
    * Target_Scale
    * BagNet_Target_Scale

## Creating jsonfile

```bash
$ python create_json.py 'dataset path(~/images)' 'phase(train or test)'
```

## Training phase
default datasets : ShanghaiTech_B

```bash
$ python main.py --root_path 'dataset path' --results_path 'specify abs path' --phase train
```

## Test phase
default datasets : ShanghaiTech_B

```bash
$ python main.py --root_path 'dataset path' --results_path 'specify results path' --load_weight True --model_path 'saved model path' --phase test
```