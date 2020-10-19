# Face-Landmark-Checker
Check the Accuracy of Face Landmark Using CNN

Training with 42 landmarks(9 contour + 33 inner) region in Labeled Faces in the Wild(LFW, http://vis-www.cs.umass.edu/lfw/) dataset, we can check the accuracy of landmark detection network.
The number of landmarks and dataset can be modified.

## 1. Training
Run main.m function and train the network of each landmark.
Image datasets used for training are stored in a "TrainingSet" folder.

## 2. Testing
Run testNetwork.m function and test
Test images are stored in a "TestSet" folder.

## 3. Usage
This network was proposed in the [paper](https://github.com/teamMint-official/Face-Landmark-Checker/blob/main/Appendix/mint_JS_2018_ICIP.pdf).
Please cite this in your publications if it helps your research

```
@inproceedings{park2018robust,
  title={Robust facial pose estimation using landmark selection method for binocular stereo vision},
  author={Park, Jaeseong and Heo, Suwoong and Lee, Kyungjune and Song, Hyewon and Lee, Sanghoon},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={186--190},
  year={2018},
  organization={IEEE}
}
```
