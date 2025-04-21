# <p align="center">The Devil Is in the Boundary: Boundary-enhanced Polyp Segmentation (FoBS [paper](https://ieeexplore.ieee.org/abstract/document/10378660))</p>
<p align="center">Zhizhe Liu<sup>1</sup>, Shuai Zheng<sup>1</sup>, Xiaoyi Sun<sup>1</sup>, Zhenfeng Zhu<sup>1</sup>, Yawei Zhao<sup>2</sup>, Xuebing Yang<sup>3</sup>, Yao Zhao<sup>1</sup></p>
<p align="center">1 Institute of Information Science, Beijing Jiaotong University</p>
<p align="center">2 Medical Big Data Research Center, Chinese PLA General Hospital</p>
<p align="center">3 Institute of Automation, Chinese Academy of Sciences</p>

![image](./img/FoBS.png)

## Datasets
Four public datasets are adopted in FoBS


* [EndoScene](https://arxiv.org/abs/1612.00799)
* [Kvasir](https://arxiv.org/abs/1911.07069)
* [ETIS](https://link.springer.com/article/10.1007/s11548-013-0926-3)
* [CVC-ColonDB](https://ieeexplore.ieee.org/document/7294676)


## To start
👉 Plz put the downloaded data or your own data in the folder below:
```
|-- FoBS
    |-- config
    |-- data
    |   |-- your own data
```

👉 You can run the single domain experiments as:

```
python  train_single_domain.py --cfg_path ./config/single_domain.yaml
```

👉 You can run the cross domain experiments as:

```
python  train_cross_domain.py --cfg_path ./config/cross_domain.yaml
```


### META
If you have any questions about this project, please feel free to drop me an email.
Zhizhe Liu -- zhzliu@bjtu.edu.cn
```
@article{liu2024devil,
  title={The devil is in the boundary: Boundary-enhanced polyp segmentation},
  author={Liu, Zhizhe and Zheng, Shuai and Sun, Xiaoyi and Zhu, Zhenfeng and Zhao, Yawei and Yang, Xuebing and Zhao, Yao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={34},
  number={7},
  pages={5414--5423},
  year={2024},
  publisher={IEEE}
}
```


## References
[1] Guo X, Yang C, Liu Y, et al. Learn to threshold: Thresholdnet with confidence-guided manifold mixup for polyp segmentation[J]. IEEE Transactions on Medical Imaging, 2020, 40(4): 1134-1146.
