# posthocOS
Code for paper: "Post-hoc Overall Survival Time Prediction from Brain MRI" [[arxiv]](https://arxiv.org/abs/2102.10765).

## Install
`pip install git+https://github.com/renato145/posthocos.git`

## Trained models
Inside `notebooks/example/trained_model.pth`

# Docker

Check: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker to use the `--gpus` flag.

```bash
# Build the image:
docker build -t posthocos .
# Run the image:
docker run --rm -it --gpus all -p 8888:8888 -v DATA_PATH:/workspace/notebooks/data posthocos:latest
```

> The Brats2019 data should be inside `DATA_PATH/raw` and the code will generate the preprocessed files in `DATA_PATH/preprocess`.
