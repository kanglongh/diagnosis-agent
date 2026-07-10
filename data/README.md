# data/

把 PU 轴承数据集下载后放到这里.

## 下载

<https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download>

下载页面底部表格, 至少下载 **K001, K002, KA01, KA05, KI01, KI05** 六个轴承.

每个轴承包含两种转速 (N09=900rpm, N15=1500rpm) 的振动数据, 文件名如 `N09_M07_F10_K001_1.mat`.

## 放好之后的结构

```
data/
├── PU/
│   ├── K001/
│   │   ├── N09_M07_F10_K001_1.mat
│   │   ├── ...
│   │   ├── N09_M07_F10_K001_20.mat
│   │   ├── N15_M07_F10_K001_1.mat
│   │   ├── ...
│   │   └── N15_M07_F10_K001_20.mat
│   ├── K002/     (同上, 20+20 个文件)
│   ├── KA01/     (同上)
│   ├── KA05/     (同上)
│   ├── KI01/     (同上)
│   └── KI05/     (同上)
└── README.md
```

## 然后

```bash
python prepare_pu.py   # CFD 特征提取 + 构造跨域任务 → pu_tasks.npz
python run.py --data pu_tasks.npz --task all
```
