# MFTPCBB

## Environment Installation
```bash

conda create -n cbbiomfp python=3.9
conda activate cbbiomfp
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html    
pip install torch-sparse==0.6.15 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-geometric   #2.6.1

# dgl
pip install dgl==1.0.1 -f https://data.dgl.ai/wheels/cu117/repo.html
pip install torchdrug
pip install hydra-core loguru rich

```
