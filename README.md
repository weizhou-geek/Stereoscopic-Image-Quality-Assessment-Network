# Stereoscopic-Image-Quality-Assessment-Network
StereoQA-Net Software release.

=======================================================================
-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2018 University of Science and Technology of China
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Immersive Media Computing Lab (IMCL) at University of Science and Technology of China 
(USTC), is acknowledged in any publication that reports research using this code. The research is to be cited
in the bibliography as:

1)  Wei Zhou, Zhibo Chen and Weiping Li, "Dual-Stream Interactive Networks for No-Reference Stereoscopic Image Quality Assessment".

2)  Wei Zhou, Zhibo Chen and Weiping Li, "StereoQA-Net Software Release", 
    URL: http://staff.ustc.edu.cn/~chenzhibo/resources.html, 2018

IN NO EVENT SHALL UNIVERSITY OF SCIENCE AND TECHNOLOGY OF CHINA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF UNIVERSITY OF SCIENCE AND TECHNOLOGY OF CHINA
HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

UNIVERSITY OF SCIENCE AND TECHNOLOGY OF CHINA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND UNIVERSITY OF SCIENCE AND TECHNOLOGY OF CHINA HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%

=======================================================================
Author  : Wei Zhou
Version : 1.0

The authors are with the Dept. of EEIS, University of Science and Technology of China, Hefei 230026, China

Kindly report any suggestions or corrections to weichou@mail.ustc.edu.cn

=======================================================================

This is a demonstration of the Stereoscpic Image Quality Assessment Network (StereoQA-Net). The algorithm is described in:

Wei Zhou, Zhibo Chen and Weiping Li, "Dual-Stream Interactive Networks for No-Reference Stereoscopic Image Quality Assessment".

You can change this program as you like and use it anywhere, but please
refer to its original source (cite our paper and our web page at
http://staff.ustc.edu.cn/~chenzhibo/resources.html, 2018).

========================================================================

Running based on Keras 

1.download test data from https://www.dropbox.com/sh/l1etoxb0s9xnjdi/AADN1mrqghB9H9xz6ewhzFV4a?dl=0 and trained model from https://www.dropbox.com/sh/42114q93y9gjz88/AAA16_v7dnyHWlh7SJwTFu_Ja?dl=0

2.test_siqa_model to predict quality score for each image patch pair

3.compute_test to obtain correlation results with MOS scores

## Citation
You may cite it in your paper. Thanks a lot.

```
@article{zhou2019dual,
  title={Dual-stream interactive networks for no-reference stereoscopic image quality assessment},
  author={Zhou, Wei and Chen, Zhibo and Li, Weiping},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={8},
  pages={3946--3958},
  year={2019},
  publisher={IEEE}
}
```
