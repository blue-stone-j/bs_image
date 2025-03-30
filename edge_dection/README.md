# edge_dection

### compile and run
```bash
# get into folder
cd source_folder
# generate empty folder to store compilation file
mkdir build
# compile configuration
cmake ..
# compile
make install
# run
./canny
```

### canny
1. 高斯滤波抑制高频噪声。
2. 计算梯度的大小和方向：直接使用差分或者 Sobel 算子实现。
3. 非极大值抑制以细化边缘：如果临近点不存在则通过差值的方式计算临近点。
4. 双阈值得到强边缘、弱边缘和平坦区域。
5. 连接弱边缘：如果周围 8 个邻接像素有一个是强边缘，则认为该弱边缘也是强边缘，否则视作孤立点抛弃。

### laplace
拉普拉斯算子是最简单的各项同性二阶微分算子，具有旋转不变性。与拉普拉斯核卷积会导致输出中出现大量噪声。 

### prewitt
在图像空间利用两个方向模板与图像进行邻域卷积来完成的，是一阶微分算子，这两个方向模板一个检测水平边缘，一个检测垂直边缘。对比其他边缘检测算子，Perwitt算子对边缘的定位精度不如Roberts算子，实现方法与Sobel算子类似，但是实现的功能差距很大，Sobel算子对边缘检测的准确性更优于Prewitt算子。

### roberts
Roberts边缘算子是一个2x2的模板，采用的是对角方向相邻的两个像素之差，在X方向和Y方向以及对角线方向上进行了边缘检测。从图像处理的实际效果来看，边缘定位较准，对噪声敏感。

### sobel
Sobel算子即可理解为同时利用了水平方向、垂直方向，以及45方向和135方向的梯度。