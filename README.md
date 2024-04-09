# AR-PointNet
paper: Three-Dimensional Point Cloud Segmentation Based on Context Feature for Sheet Metal Part Boundary Recognition. in IEEE Transactions on Instrumentation and Measurement

code: come soon

## Abstract
Point cloud is widely available in the manufacturing system with the continuous development of 3-D sensors. Accurate point cloud segmentation can automatically identify different components during the manufacturing process, which is essential to the quality assurance of the final product. However, the existing methods for point cloud segmentation fail to accurately identify the boundary of each region, which decays the quality of many manufacturing operations (i.e., welding). In this article, a point cloud segmentation model, Attention Recurrent PointNet (AR-PointNet), is proposed to improve the segmentation accuracy, especially on the regions’ boundaries. More specifically, the PointNet is used as the backbone, and the novel channel and spatial statistical feature (SSF) attention modules are proposed and combined to enhance the features along the regions’ edges. In addition, context features among different regions in the point cloud are further extracted to consider the interactions among components. The experiments are conducted on the ShapeNet dataset and the real-scanned dataset of manufactured products, respectively. The results demonstrated that: 1) on the ShapeNet dataset, the proposed method outperforms the state-of-the-art segmentation models. The mean intersection-over-union (mIoU) is selected as the evaluation metric to indicate the segmentation accuracy. The proposed method improves mIoU by 2.3% compared with its backbone (PointNet) and 2) on the simulated and real-scanned sheet metal part, it successfully recognizes the boundaries of components in the sheet metal part and significantly improves the recognition accuracy. Boundary mean accuracy (BMA) is defined as the evaluation metric to indicate the boundary recognition accuracy. Results of the simulated data and the real-scanned data show that the proposed method improves the BMA by 4.4% and 3.6%, respectively, compared with its backbone (PointNet).

## Citation
If you find our work useful in your research, please consider citing:

```
@ARTICLE{10113785,
  author={Li, Yanzheng and Wang, Yinan and Liu, Yinhua},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Three-Dimensional Point Cloud Segmentation Based on Context Feature for Sheet Metal Part Boundary Recognition}, 
  year={2023},
  volume={72},
  number={},
  pages={1-10},
  doi={10.1109/TIM.2023.3272047}
}
```

## Installation
The code has been tested on following environment

```
Windows 10
python 3.6
CUDA 11.0
torch 1.10.0
```
