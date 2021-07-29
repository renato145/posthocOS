# posthocOS
Code for paper: ["Post-hoc Overall Survival Time Prediction from Brain MRI"](https://ieeexplore.ieee.org/document/9433877)
[[arxiv]](https://arxiv.org/abs/2102.10765).

## Install
`pip install git+https://github.com/renato145/posthocos.git`

## Trained models
Pytorch model: `notebooks/example/trained_model.pth`

## Docker

Check: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to use the `--gpus` flag.

```bash
# Build the image:
docker build -t posthocos .
# Run the image:
docker run --rm -it --gpus all -p 8888:8888 -v DATA_PATH:/workspace/notebooks/data posthocos:latest
```

> The Brats2019 data should be inside `DATA_PATH/raw` and the code will generate the preprocessed files in `DATA_PATH/preprocess`.

## Citation

Please use the following bibtex entry:
```bibtex
@INPROCEEDINGS{hermoza2021posthoc,
  author={Hermoza, Renato and Maicas, Gabriel and Nascimento, Jacinto C. and Carneiro, Gustavo},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)}, 
  title={Post-Hoc Overall Survival Time Prediction From Brain MRI}, 
  year={2021},
  volume={},
  number={},
  pages={1476-1480},
  doi={10.1109/ISBI48211.2021.9433877}}
```
