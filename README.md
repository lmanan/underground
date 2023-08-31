# underground
Instance Segmentation without GT


## Install Environment

```
conda create -n underground-env python
conda activate underground-env
pip install numpy matplotlib scikit-image jupyterlab jupyter gunpowder
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
python -m pip install 'cellpose[gui]'
python -m pip install "napari[all]"
```
