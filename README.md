# Zebrafish_Heart_Segmentation_3D_FYP_AC - Using keras and TF.Keras
 All the code and files required for completion of segmentation of 3D images of zebrafish emnryo hearts and the analysis of the results.

Pre-trained weights for each of the 3D backbones, the modules for loading the weights and some of the code for the backbone models was taken from [classification_models_3D](https://github.com/ZFTurbo/classification_models_3D.git) repo by [@ZFTurbo]

### Segmentation Architectures Used:
- [UNet](https://arxiv.org/abs/1505.04597)
- [AttentionUNet](https://arxiv.org/abs/1804.03999)
- [AttentionResUNet](https://arxiv.org/abs/2011.14302)

### Pre-trained Classification Backbones Used:
- [VGG16](https://arxiv.org/abs/1409.1556)
- [ResNet-18, 34, 50](https://arxiv.org/abs/1512.03385)

## Requirements:
- tensorflow
- keras
- patchify