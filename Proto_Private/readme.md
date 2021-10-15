# Official implementation for **PCT for source-private adaptation**

### [**We modify the code from SHOT. Please also cite the original author if you intend to use this code.**](https://github.com/tim-learn/SHOT)


### Prerequisites:
- python == 3.6.8
- pytorch ==1.1.0
- torchvision == 0.3.0
- numpy, scipy, sklearn, PIL, argparse, tqdm

### Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [Office-Caltech](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz) from the official websites, and modify the path of images in each '.txt' under the folder './object/data/'.


### Training:

Please refer to ***run.sh*** for all the settings for different methods and scenarios. We provide examples for Office datasets.

### Citation

If you find this framework useful for your research, please cite the original author of the codebase as well as our paper.

### PCT
> @inproceedings{tanwisuth2021prototype,  
>  title={A Prototype-Oriented Framework for Unsupervised Domain Adaptation},  
>  author={Korawat Tanwisuth and Xinjie Fan and Huangjie Zheng and Shujian Zhang and Hao Zhang and Bo Chen and Mingyuan Zhou},  
> booktitle = {NeurIPS 2021: Neural Information Processing Systems},   
> month={Dec.},  
> Note = {(the first three authors contributed equally)},  
> year = {2021}  
> }  

### SHOT
> @inproceedings{liang2020shot,  
>  &nbsp; &nbsp;  title={Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation},  
>  &nbsp; &nbsp;  author={Liang, Jian and Hu, Dapeng and Feng, Jiashi},  
>  &nbsp; &nbsp;  booktitle={International Conference on Machine Learning (ICML)},  
>  &nbsp; &nbsp;  pages={6028--6039},  
>  &nbsp; &nbsp;  month = {July 13--18},  
>  &nbsp; &nbsp;  year={2020}  
> }


