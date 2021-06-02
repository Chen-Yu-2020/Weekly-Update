# Weekly-Update
My weekly Update

# Convex Optimization

###### Using First-order methods

1. Gradient descent

   ​	其实所有的利用一阶导数进行下降的都可以叫梯度下降法，这里介绍的梯度下降法实际上是牛顿法。利用对目标函数的二阶泰勒展开进行近似，求取平稳点得到下一个迭代点的位置。

$$
f(z) = f(x)+\nabla f(x)^T(z-x)+\frac{1}{2}(z-x)\nabla^2f(x)(z-x)
$$

$$
\nabla f(z) = \nabla f(x)+\nabla^2f(x)(z-x)=0
$$

$$
z = x - [\nabla^2f(x)]^{-1}\nabla f(x)
$$

​		<img src="C:\Users\10272\AppData\Roaming\Typora\typora-user-images\image-20210601182110785.png" alt="image-20210601182110785" style="zoom:33%;" />

​		Q1：为什么用二阶泰勒展开？
​		从图上可以看出，如果是一阶近似，没有最优点，无法进行迭代。二阶泰勒展开中的二阶项限制下一个迭代点在当前迭代点附近，这个性质在近端梯度法中会用到。
​		

​		除了方向以外，还需要考虑下降的步长。网上的大部分资料都是把负梯度当作下降方向，直接对步长进行一维线性搜索，最常见的是backline search:
$$
\begin{align}
&0<\alpha \leq \frac{1}{2},0<\beta <1\\
&while \qquad f(x+t\Delta x)>f(x)+\alpha t\Delta x\nabla f(x) \\
&t = \beta t
\end{align}
$$
​		可以看到如果把f(x)在t=0处做一阶泰勒展开，得到以t为横轴，f(x+t\Delta x)为纵轴的在t=0处的近似一维函数。
$$
f(x+t\Delta x)=f(x)+\Delta x \nabla f(x)t+o(t^2)
$$
​		把其中的\Delta x 换成负梯度后，搜索条件变为：
$$
f(x-t\Delta f(x))>f(x)-\alpha t ||f(x)||_2^2
$$

<img src="C:\Users\10272\AppData\Roaming\Typora\typora-user-images\image-20210601131138899.png" alt="image-20210601131138899" style="zoom:33%;" />


2. Sub gradient

   ​	我个人认为次梯度是用于描述不可微分的目标函数的梯度的一种数学上的工具。其利用的是凸函数的一阶条件：
   $$
   f(z)\geq f(x)+\nabla f(x)^T(z-x)
   $$
   假如f(x)不可微，则用g(x)代替f(x)的梯度：（这里不可求导的概念有点模糊，教材中用的是not smooth这样的字眼）
   $$
   f(z)\geq f(x)+g(x)^T(z-x)
   $$
   符合条件的g(x)则成为次梯度。次梯度的性质很多，其中一点是最优解的判别条件，即在最优点处，0必然包含在次梯度集合中。

   

3. 次梯度下降

   ​	次梯度下降方法与梯度下降实现类似，即把\Delta f(x)换成g(x)，进行梯度下降。同时次梯度下降的步长是提前设定的不断减小的序列，并不按照每轮迭代找最优步长；但次梯度下降法并不保证每次目标函数值下降。

   

4. 近端梯度法

   ​	近端梯度法的思想是将目标函数分为可微和不可微的部分，用可微部分的二阶泰勒展开的极小值点位置作为限制，寻找不可微部分的极小值点：
   $$
   min \, f(x)=g(x)+h(x) \\
   min \, f(z) = g(x)+\nabla g(x)^T(z-x)+\frac{1}{2}(z-x)^T\nabla^2g(x)(z-x)+h(z) \\
   min \, f(z) = \frac{1}{2}\nabla^2g(x)||(x-[\nabla ^2g(x)]^{-1}\nabla g(x))-z||_2^2+h(z) \\
   if \, set\, \nabla^2g(x)=\frac{1}{t} \, then \\
   min \, f(z)=\frac{1}{2t}||(x-t\nabla g(x))-z||_2^2 + h(z)
   $$
   ​	以写成类似LASSO的形式为例子：
   $$
   {\underset{\beta }{\operatorname{arg\,min}}}\,\frac{1}{2}||y-X\beta ||_2^2+\lambda||\beta||_1
   $$
   ​	这个问题第一部分为二范数显然可微为g(x)，第二部分为一范数不可微为h(x)。对于可微的第一部分，其实理论上可以求得在其极小值点为$ X^T(y-X\beta)=0 $ (即最小二乘解)，根据近端梯度法，其优化目标可以写成:
   $$
   \underset{z}{\operatorname{arg\,min}}\,\frac{1}{2t}||z-(X^TX)^{-1}X^Ty||^2_2+\lambda||z||_1
   $$
   ​	但实际算法中并不会采用一次性迭代到最小值的方法，而是利用其一阶梯度及backline search的方法，配合近端梯度算子进行求解：
   $$
   \nabla \frac{1}{2}||y-X\beta||^2_2=-X^T(y-X\beta)
   $$

   $$
   \underset{z}{arg\,min}\,\frac{1}{2t}||(\beta +tX^T(y-X\beta))-z||_2^2+\lambda||z||_1
   $$

   ​	得到优化目标后，定义近端梯度算子：
   $$
   prox_t(x)=\underset{z}{arg\,min}\,\frac{1}{2t}||x-z||_2^2+\lambda||z||_1
   $$
   ​	优化目标可重写为$ prox_t(\beta+tX^T(y-X\beta)) $, 由于近端梯度算子有解析解：
   $$
   prox_t(x)=\left\{
   \begin{matrix}
   x-\lambda, \quad &if \, x>\lambda \\ 
   0,\quad &if \, -\lambda \leq x \leq \lambda \\
   x+\lambda,\quad &if \,x<-\lambda
   \end{matrix}
   \right.
   $$
   ​	由此即可对LASSO问题进行近端梯度求解，此求解算法称为iterative shrinkage-thresholding algorithm (ISTA) 也有人叫 iterative soft-thresholding algorithm

   <img src="C:\Users\10272\AppData\Roaming\Typora\typora-user-images\image-20210601182010520.png" alt="image-20210601182010520" style="zoom:33%;" />

5. Sochastical gradient descent（SGD）

   ​	SGD，随机梯度下降，是deep learning中常用的方法。假如目标函数是由多个子目标函数相加构成$ f(x)=\frac{1}{m}\sum_{i=1}^{m} f_i(x)$, 其利用梯度下降的优化迭代式可以写成：
   $$
   x^k=x^{k-1}-t_k\nabla f_{i_k}(x_{k-1}),k=1,2,3...
   $$
   ​	每一轮迭代对一个子目标进行梯度下降，多轮迭代后其数学期望与对于总目标函数进行梯度下降法相等。

   ​	除了每次对一个子目标进行梯度下降，也可以对多个子目标的加和梯度进行下降，成为mini-batch。梯度下降，随机梯度下降，mini-batch的收敛速度对比结果如下。

   <img src="C:\Users\10272\AppData\Roaming\Typora\typora-user-images\image-20210601184114470.png" alt="image-20210601184114470" style="zoom:33%;" />

   ​	优点：节约计算空间和计算时间，不需要计算总目标函数的梯度

   ​	缺点：收敛速度很慢

#### ISTA实现及感想

​		磁共振重建问题的数学模型：
$$
\underset{x}{arg\,min}\,||y-F_ux||_2^2+\lambda||\Psi x||_1=g(x)+h(x)
$$
​		$$ y $$表示k空间实际采集到的数据（欠采样），$ x $ 表示待重建的图像$$ F_u $$表示FFT+mask，$ \Psi $表示稀疏变换。

​		观察该目标函数，显然第一部分可微，第二部分不可微分。因此可以使用近端梯度法：
$$
\underset{z}{arg\,min}\,||x+F_u^*(y-F_ux)-z||^2_2+\lambda||\Psi z||_1
$$
​		但由于该形式不符合LASSO，不能直接用soft-thresholding，因此对其进行变换，$ u = \Psi z $，重写得：
$$
\underset{u}{arg\,min}\, || \Psi (x+F_u^*(y-F_ux))-u||_2^2 + \lambda||u||_1
$$
​		该问题有解析解：$ u=softthresh(\Psi (x+F_u^*(y-F_ux))),\lambda) $ 

​		迭代流程如下：

​				$ u_0 = init;  x_0 = \Psi ^* u_0; \lambda = 0.0005; $

​				$ x_k = x_{k-1} + F_u^*(y-F_u x_{k-1}); $

​				$ u_k = softthresh(\Psi x_k, \lambda); $

```matlab
%% ISTA
% object = ||F*u-y||^2_2 + ||u||_1 = g(u) + h(u)
% gradient_g = F'*(F*u-y)
% gradient_g2 = I
% u 为稀疏域，y为k空间，F为从稀疏域到k空间的变换
imgdata = full_img(:,:,1);
kdata = coil_kspace(:,:,1);
W = Wavelet;
y = kdata .* mask_PE;
dc_img = ifft2(ifftshift(y));
u = W * dc_img;
iter = 0; lambda = 0.0005;
figure,
subplot(131); imshow(abs(imgdata),[]);
subplot(132); imshow(abs(dc_img),[]);
while iter < 80
    temp = max(abs(u(:)));
    u = u / temp;
    u = (abs(u) > lambda) .* (abs(u)-lambda) ./ abs(u) .* u; % softthresholding
    u = u * temp;
    y1 = fftshift(fft2((W'* u)));
    g = W * (ifft2(ifftshift((y1-y).* mask_PE))); % gradient
    u = u - g; 
    iter = iter + 1;
    k_error = (fftshift(fft2(W' * u))-y) .* mask_PE;
    error(iter) = norm(abs(k_error(:)),2) + lambda * norm(abs(u),1); % loss
end
subplot(133) ; imshow(abs(W' * u));
figure, plot(error);
```

#### ISTA不能准确重建Phase

![image-20210602233731381](C:\Users\10272\AppData\Roaming\Typora\typora-user-images\image-20210602233731381.png)



