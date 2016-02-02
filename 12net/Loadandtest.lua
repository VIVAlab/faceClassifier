require 'nn'
require 'nnx'
require 'image'
require 'torch'
local filename ='/Users/wbh/cnn/resize/img/2.jpg'
--local filename ='/Users/wbh/cnn/test/c_faces/pic00009.jpg'
--local filename ='/Users/wbh/cnn/test/nonfaces/40009.jpg'
--local filename = '/Users/wbh/cnn/src/3.jpg'
local im =  image.load(filename):float()
im=image.scale(im,12,12)
local imean=im:mean()
local istd=im:std()
im:add(-imean)
im:div(istd)
torch.setdefaulttensortype('torch.FloatTensor')
-- network1 = torch.load('/home/binghao/cnn/model.net')
network1 = torch.load('/Users/wbh/cnn/model.net')
net12 = nn.Sequential()
net12:add(network1.modules[1])
net12:add(network1.modules[2])
net12:add(network1.modules[3])
net12:add(network1.modules[4])
net12:add(network1.modules[5])
net12:add(network1.modules[6])
net12:add(network1.modules[7])
net12:add(network1.modules[8])
--classifier1=nn.Sequential()
--classifier1:add(network1.modules[6])
--classifier1:add(network1.modules[7])
--classifier1:add(network1.modules[8])
--classifier=nn.SpatialClassifier(classifier1)
--net12:add(classifier)
im=image.rgb2y(im)
-- print(im)
y=net12:forward(im)
print(y)
y = torch.exp(y)
print(y[1])
print(y[2])
-- torch.save('test3.dat',y)
-- y=torch.load('test1.dat')
