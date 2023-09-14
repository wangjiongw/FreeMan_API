# FreeMan: Towards Benchmarking 3D Human Pose Estimation in the Wild
This is the Official Repository for FreeMan dataset

<p align="left">
    <font size='6'>
    <a href="https://wangjiongw.github.io/freeman" target="_blank">🌏 Project Page</a> • 
      <a href="https://wangjiongw.github.io/freeman/download.html" target="_blank">🙋‍♂️ Request</a> • 
      <a href="https://arxiv.org/abs/2309.05073" target="_blank">📄 Paper </a> • 
      <a href="https://www.youtube.com/watch?v=g2h1YW-3n5k" target="_blank">▶️ YouTube </a> • 
      <a href="https://github.com/wangjiongw/FreeMan_API" target="_blank">🖥️ Code </a>
    </font>
</p>

![](./figs/Intro.png)

## News
[2023-09-07]
[Project page](https://wangjiongw.github.io/freeman) updated! Details & download methods are presented.

[2023-06-15]
Hi! We are almost there! Data are uploading to cloud server. Please sign this [Form](https://forms.gle/XN3UE6ZqPYyQG76Y7) for latest updates.

## Dataset Structure
After downloading datasets, file structure are organized as follows.
```
    FreeMan
        ├── 30FPS
        │   ├── bbox2d
        │   │   ├── yyyymmdd_xxxxxxxx_subjNN.npy
        │   │   └── ...
        │   ├── keypoints2d
        │   │   ├── yyyymmdd_xxxxxxxx_subjNN.npy
        │   │   └── ...
        │   ├── keypoints3d
        │   │   ├── yyyymmdd_xxxxxxxx_subjNN.npy
        │   │   └── ...
        │   ├── motions
        │   │   ├── yyyymmdd_xxxxxxxx_subjNN_view0.npy
        │   │   ├── ...
        │   │   ├── yyyymmdd_xxxxxxxx_subjNN_view8.npy
        │   │   └── ...
        │   ├── cameras
        │   │   ├── yyyymmdd_xxxxxxxx_subjNN.json
        │   │   └── ...
        │   └── videos
        │        ├── yyyymmdd_xxxxxxxx_subjNN
        │        │   ├── cameras.json
        │        │   ├── chessboard.pkl
        │        │   └── vframes
        │        │       ├── c01.mp4
        │        │       ├── ...
        │        │       └── c08.mp4
        │        ├── ...
        ├── 60FPS
        │   ├── ...
```

## Usage
```python
from freeman_loader import FreeMan
# Initialize dataset
freeman = FreeMan(base_dir="YOUR PATH", fps="30 or 60")

# Load video frame
video_frames = FreeMan.load_frames(freeman.get_video_path("SESSION_ID", "CAM ID"))

# Load keypoints
kpts2d = FreeMan.load_keypoints2d(freeman.keypoints2d_dir, "SESSION_ID")
kpts3d = FreeMan.load_keypoints3d(freeman.keypoints3d_dir, "SESSION_ID")

```

## Citation

If you find FreeMan helpful and used in your project, please cite our paper.
```
@article{wang2023freeman,
  title={FreeMan: Towards Benchmarking 3D Human Pose Estimation in the Wild},
  author={Wang, Jiong and Yang, Fengyu and Gou, Wenbo and Li, Bingliang and Yan, Danqi and Zeng, Ailing and Gao, Yijun and Wang, Junle and Zhang, Ruimao},
  journal={arXiv preprint arXiv:2309.05073},
  year={2023}
}
```

## License & Ackowledgement

This project and FreeMan dataset uses lisence of CC BY NC SA 4.0.

Great appreciation to all volunteers participated in FreeMan.

Thanks to great work of [AIST++](https://github.com/google/aistplusplus_api).
