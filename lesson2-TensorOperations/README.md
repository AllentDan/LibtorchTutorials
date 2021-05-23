## Tensor initialization
Most APIs of libtorch (pytorch c++) are consistent with pytorch, so the initialization of tensors in libtorch is similar to that in pytorch. This article introduces four initialization methods required for deep learning programming.

**The first** is the initialization of a fixed size and value.
```cpp
//Common fixed value initialization methods
auto b = torch::zeros({3,4});
b = torch::ones({3,4});
b= torch::eye(4);
b = torch::full({3,4},10);
b = torch::tensor({33,22,11});
```
In pytorch, [] is used to indicate size, while in cpp, {} is used. Function zeros produces a tensor with all zeros. Function ones produces a tensor with all 1s. Function eye generates an identity matrix tensor. Function full produces a tensor of the specified value and size. Function torch::tensor({}) can also generate tensors, the effect is the same as pytorch's torch.Tensor([]) or torch.tensor([]).

**The second**, fixed size, random value initialization method:
```cpp
//Random initialization
auto r = torch::rand({3,4});
r = torch::randn({3, 4});
r = torch::randint(0, 4,{3,3});
```
rand generates a random value between 0-1, randn takes the random value of the normal distribution N(0,1), and randint takes the random integer value of [min,max).

**The third**, converted from other data types in C++
```cpp
int aa[10] = {3,4,6};
std::vector<float> aaaa = {3,4,6};
auto aaaaa = torch::from_blob(aa,{3},torch::kFloat);
auto aaa = torch::from_blob(aaaa.data(),{3},torch::kFloat);
```
Pytorch accepts data from other data types such as numpy and list into tensors. Libtorch can also accept other data pointers, which can be converted through the from_blob function. This method is often used in deployment. If the image is loaded by opencv, the image pointer can be converted into a tensor through from_blob.

**The fourth** type is initializing tensor according to the existing tensor:
```cpp
auto b = torch::zeros({3,4});
auto d = torch::Tensor(b);
d = torch::zeros_like(b);
d = torch::ones_like(b);
d = torch::rand_like(b,torch::kFloat);
d = b.clone();
```
Here, auto d = torch::Tensor(b) is equivalent to auto d = b. The tensor d initialized by them is affected by the original tensor b. When the value in b changes, d will also change, but if b deformed, d will not follow the deformation and still maintain the initial shape. This performance is called shallow copy. Functions zeros_like and ones_like, as the name suggests, will generate 0 tensor and 1 tensor with the same shape as the original tensor b, and randlike is the alike. The last clone function completely copied b into a new tensor d, the change of the original tensor b will not affect d, which is called a deep copy.

## Tensor deformation
Torch changes the shape of the tensor, does not change the tensor's content pointed to by the data pointer, but changes the way the tensor is accessed. The deformation method of libtorch is the same as that of pytorch, and there are commonly used deformations such as view, transpose, reshape, and permute.

```cpp
auto b = torch::full({10},3);
b.view({1, 2,-1});
std::cout<<b;
b = b.view({1, 2,-1});
std::cout<<b;
auto c = b.transpose(0,1);
std::cout<<c;
auto d = b.reshape({1,1,-1});
std::cout<<d;
auto e = b.permute({1,0,2});
std::cout<<e;
```
.view is not an inplace operation, you need to add =. There is not much to say about the deformation operation if you are familiar with pytorch. There are also squeeze and unsqueeze operations, which are also the same as pytorch.
## Tensor cutout
To intercept the tensor by index, the code is as follows
```cpp
auto b = torch::rand({10,3,28,28});//BxCxHxW
std::cout<<b[0].sizes();//0th picture
std::cout<<b[0][0].sizes();//0th picture, 0th channel
std::cout<<b[0][0][0].sizes();//0th picture, 0th channel, 0th row pixels
std::cout<<b[0][0][0][0].sizes();//0th picture, 0th channel, 0th row, 0th column pixels
```
Apart from [], there are narrow，select，index，index_select functions.
```cpp
std::cout<<b.index_select(0,torch::tensor({0, 3, 3})).sizes();//choose 0th dimension at 0,3,3 to form a tensor of [3,3,28,28]
std::cout<<b.index_select(1,torch::tensor({0,2})).sizes(); //choose 1th dimension at 0 and 2 to form a tensor of[10, 2, 28, 28]
std::cout<<b.index_select(2,torch::arange(0,8)).sizes(); //choose all the pictures' first 8 rows [10, 3, 8, 28]
std::cout<<b.narrow(1,0,2).sizes();//choose 1th dimension, from 0, cutting out a lenth of 2, [10, 2, 28, 28]
std::cout<<b.select(3,2).sizes();//select the second tensor of the third dimension, that is, the tensor composed of the second row of all pictures [10, 3, 28]
```
The index needs to be especially described. In pytorch, it is easy to filter tensors through the mask Mask directly Tensor[Mask]. However, it cannot be used directly in C++ It needs to be implemented by the index function. The code is as follows:
```cpp
auto c = torch::randn({3,4});
auto mask = torch::zeros({3,4});
mask[0][0] = 1;
std::cout<<c;
std::cout<<c.index({mask.to(torch::kBool)});
```
Some netizens asked questions: the tensor from index is the result of deep copy, that is, a new tensor obtained. Then how to modify the value pointed to by the mask of the original tensor. Seach torch's apis and find that there is also a index_put_ function can directly place the specified tensor or constant. The combination of index_put_ and index functions can achieve this task.
```cpp
auto c = torch::randn({ 3,4 });
auto mask = torch::zeros({ 3,4 });
mask[0][0] = 1;
mask[0][2] = 1;
std::cout << c;
std::cout << c.index({ mask.to(torch::kBool) });
std::cout << c.index_put_({ mask.to(torch::kBool) }, c.index({ mask.to(torch::kBool) })+1.5);
std::cout << c;
```
In addition, there is a common way of accessing numbers in python, tensor[:,0::4], which is a way of accessing numbers in the 1th dimension with a starting position of 0 and an interval of 4, which is directly implemented in c++ with the slice function.
## Inter-tensor operations
Concate and stack
```cpp
auto b = torch::ones({3,4});
auto c = torch::zeros({3,4});
auto cat = torch::cat({b,c},1);//1 refers to 1th dim, output a tensor of shape [3,8]
auto stack = torch::stack({b,c},1);//1refers to 1th dim, output a tensor of shape [3,2,4]
std::cout<<b<<c<<cat<<stack;
```
At this point, readers will find that it is much easier to use libtorch if you master the changes from [] in pytorch to {} in libtorch. Most operations can be directly migrated.

It can also applied for the four arithmetic operations. Just use * and / to multiply and divide the corresponding elements, and you can also use .mul and .div. Matrix multiplication uses .mm, adding batch is .bmm.
```cpp
auto b = torch::rand({3,4});
auto c = torch::rand({3,4});
std::cout<<b<<c<<b*c<<b/c<<b.mm(c.t());
```
Some other operations like clamp, min, max are similar to pytorch, and can be explored by yourselves according to the above method.