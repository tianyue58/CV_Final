# CV-Spring23-FDU

This is the project of DATA130051 Computer Vision.

## Task 1

Run it in Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mr8yziuhmGQoFeQspqY-ks3Ga_iU8Lnr?usp=sharing)

## Task 2

In the second part of the project, we use vision transformer to build classifiers for CIFAR-100, and try to use augmentation method, Cut-Mix, to optimize the model. And we finally get a top-1 accuracy of 90.78 percent and top-5 accuracy of 99.07 percent using transformer. 

The usages of the codes like training and testing processes are shown in the corresponding folders.

### Augmentation

<img alt="Unaugmented" src="https://github.com/tianyue58/CV_Final/assets/77108843/6cfc91f0-fd41-4957-a9d0-322bcd9e2655">

<img alt="Cut-Mix" src="https://github.com/tianyue58/CV_Final/assets/77108843/e4b71f7e-686c-4d89-8d64-f0be50d04aaa">


### Experiments

|  Dataset  |                                             Neural Network                                              | Augmentation | Accuracy | Accuracy Top 5 |
| :-------: | :-----------------------------------------------------------------------------------------------------: | :----------: | :------: | :------------: |
| CIFAR-100 | [EfficientNetB7](https://drive.google.com/file/d/1e58zfXlcOxf74zKYt3RVzxzfURkLg7lZ/view?usp=share_link) |   Cut-Mix    |  88.35%  |     98.76%     |
| CIFAR-100 | [VitB16-Cut-Mix](https://drive.google.com/file/d/1qC3C4FZ721rVoAat739uPHGhm1bP7bZk/view?usp=share_link) |   Cut-Mix    |  90.78%  |     99.07%     |

