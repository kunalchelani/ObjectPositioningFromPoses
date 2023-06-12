# ObjectPositioningFromPoses
This is the Github page for the CVPR 2023 paper - "Privacy Preserving Representations are not Enough: Recovering Scene Content from Camera Poses" <br>
Authors - Kunal Chelani<sup>1</sup>, Torsten Sattler<sup>2</sup>, Fredrik Kahl<sup>1</sup>, Zuzana Kukelova<sup>3</sup> <br>
<sup>1</sup> Chalmers University of Technology <br>
<sup>2</sup> Czech Institute of Informatics, Robotics and Cybernetics, Czech Technical University in Prague <br>
<sup>3</sup> Visual Recognition Group, Faculty of Electrical Engineering, Czech Technical University in Prague <br>
  
### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chelani_Privacy-Preserving_Representations_Are_Not_Enough_Recovering_Scene_Content_From_Camera_CVPR_2023_paper.pdf) | [Video](https://www.youtube.com/watch?v=8qmkkrMayZo&t=2s)

Please use the following to cite our work if the ideas or code is useful for your research:

```
@InProceedings{Chelani_2023_CVPR,
    author    = {Chelani, Kunal and Sattler, Torsten and Kahl, Fredrik and Kukelova, Zuzana},
    title     = {Privacy-Preserving Representations Are Not Enough: Recovering Scene Content From Camera Poses},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {13132-13141}
}
```

This repository contains the proof-of-concept pipeline emulating the attack on a localization server using camera poses. Images of an object captured from different viewpoints are used as quesries for localization on the server and the poses obtained are used to infer the position of the object, in the server scene. Utilities for visualizing the resulting attack are also included. We use [HLoc](https://github.com/cvg/Hierarchical-Localization) to emulate the localization server.
