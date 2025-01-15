from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
#from torchsummary import summary
import torch.nn.functional as F

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
#from efficientnet_pytorch import EfficientNet

import timm
from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import swin_transformer


import csv
import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

ResultSave_path='record_fourkener_loss3_new_koniq.txt'

torch.backends.cudnn.benchmark = True

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            #print(img_name)
            #print('ok')
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2, padding=2)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        #print(x.shape)

        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        
        return x

class cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=5, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        _x = x1
        B, N, C = x1.shape
        qkv = self.qkv1(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        _q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = self.qkv2(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, _k, _v = q.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + _x
        return x
    
    
class BaselineModel2(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(BaselineModel2, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, 2)
        self.bn3 = nn.BatchNorm1d(2)              #add norm
        #self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        #out = self.drop1(out)
        out = self.fc2(out)
        #print(out.shape)

        out = self.bn2(out)
        out = self.relu2(out)
        #out = self.drop2(out)
        more_out = out
        out = self.fc3(out)
        out = self.bn3(out)
        #out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out
    
    
class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        
        self.fc3 = nn.Linear(512, 125)
        self.bn3 = nn.BatchNorm1d(125)
        self.relu3 = nn.PReLU()
        self.drop3 = nn.Dropout(p=self.drop_prob)
        
        #self.fc4 = nn.Linear(25, num_classes)
        #self.bn4 = nn.BatchNorm1d(num_classes)              #add norm
        #self.sig = nn.Sigmoid()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        ex_out = out
        out = self.bn3(out)
        out = self.relu3(out)
        return out

# Gradient Reversal Layer
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

GRL = GradientReverseLayer.apply


class Net1(nn.Module):
    def __init__(self, net1, linear):
        super(Net1, self).__init__()
        self.NET1 = net1
        
        self.Linear = linear

    def forward(self, x1):

        x1 = self.NET1(x1)
        #print(x1.size)
        
        x, ex_out = self.Linear(x1)
        
        return x, x1

class extratcter_map(nn.Module):
    def __init__(self, extratcter, linear, cnn, c_att):

        super(extratcter_map, self).__init__()
        self.ex_net = extratcter
        self.Linear = linear
        self.onecnn = cnn
        self.atten = c_att
        
        self.fc3 = nn.Linear(375, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.PReLU()
        
        self.fc4 = nn.Linear(128, 1)
        self.bn4 = nn.BatchNorm1d(1)
        self.relu4 = nn.PReLU()

    def forward(self, x):
        ex_out = 0
        x_feature = self.ex_net(x)
        
        x_linear = self.Linear(x_feature)
        #print('fea', x_linear.shape)
        #x_linear1 = x_linear.unsqueeze(1)
        
        x_cnn = self.onecnn(x_feature)
        x_cnn1 = x_cnn 
        x_cnn = x_cnn.squeeze(1)
        #print('fea', x_cnn.shape)
        
        x_atten = self.atten(x_cnn1, x_linear)
        x_atten = x_atten.squeeze(1)
        #print('fea', x_atten.shape)
        feature = torch.cat((x_linear, x_cnn, x_atten), dim=1)
        
        feature = self.fc3(feature)
        feature = self.bn3(feature)
        feature = self.relu3(feature)
        
        feature = self.fc4(feature)
        feature = self.bn4(feature)
        feature = self.relu4(feature)
        #print(feature.shape)
        
        return feature, x_feature 
    
    
class Net(nn.Module):
    def __init__(self , net1, linear2):
        super(Net, self).__init__()
        self.feature_extractor = net1
        #self.quality_analysis = linear1
        self.domain_classifier = linear2

    def forward(self, x1, alpha=0.0):

        features, x_feature = self.feature_extractor(x1)
        
        #domain_output = self.domain_classifier(GRL(features))
        domain_output = self.domain_classifier(GRL(x_feature))
        #quality_output = self.quality_analysis(features)
        
        return features, domain_output


def computeSpearman(dataloader_valid1, dataloader_valid2, model):
    ratings1 = []
    predictions1 = []
    ratings2 = []
    predictions2 = []
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    with torch.no_grad():
        cum_loss = 0
        for data1, data2 in zip(dataloader_valid1, dataloader_valid2):
            inputs1 = data1['image']
            batch_size1 = inputs1.size()[0]
            labels1 = data1['rating'].view(batch_size1, -1)
            # labels = labels / 10.0
            inputs2 = data2['image']
            batch_size2 = inputs2.size()[0]
            labels2 = data2['rating'].view(batch_size2, -1)
           
            if use_gpu:
                try:
                    inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                    inputs2, labels2 = Variable(inputs2.float().cuda()), Variable(labels2.float().cuda())
                except:
                    print(inputs1, labels1, inputs2, labels2)
            else:
                inputs1, labels1 = Variable(inputs1), Variable(labels1)
                inputs2, labels2 = Variable(inputs2), Variable(labels2)

            #x, share_feature, x_classify = model_help(inputs2)

            #outputs_a, k1, k11, k2, k3 = model(inputs1, inputs1, inputs3, share_feature)
            label_output1, domain_output1 = model(inputs1, 0.0)
            label_output2, domain_output2 = model(inputs2, 0.0)
            
            ratings1.append(labels1.float())
            predictions1.append(label_output1.float())
            
            ratings2.append(labels2.float())
            predictions2.append(label_output2.float())
            
            _1, predicted1 = torch.max(domain_output1, 1)
            predicted1 = predicted1.squeeze()
            _2, predicted2 = torch.max(domain_output2, 1)
            predicted2 = predicted2.squeeze()
            
            total1 += labels1.size(0)
            total2 += labels2.size(0)
            do_label1 = torch.zeros(labels1.size(0), dtype=torch.float).squeeze().cuda()
            do_label2 = torch.ones(labels2.size(0), dtype=torch.float).squeeze().cuda()
            correct1 += (predicted1 == do_label1).sum().item()
            correct2 += (predicted2 == do_label2).sum().item()
            

    ratings_i1 = np.vstack([r.detach().cpu().numpy() for r in ratings1])
    predictions_i1 = np.vstack([p.detach().cpu().numpy() for p in predictions1])
    ratings_i2 = np.vstack([r.detach().cpu().numpy() for r in ratings2])
    predictions_i2 = np.vstack([p.detach().cpu().numpy() for p in predictions2])
    a1 = ratings_i1[:,0]
    b1 = predictions_i1[:,0]
    a2 = ratings_i2[:,0]
    b2 = predictions_i2[:,0]
    sp1 = spearmanr(a1, b1)[0]
    pl1 = pearsonr(a1,b1)[0]
    sp2 = spearmanr(a2, b2)[0]
    pl2 = pearsonr(a2,b2)[0]
    
    accuracy1 = 100 * correct1 / total1
    accuracy2 = 100 * correct2 / total2
    
    return sp1, pl1, sp2, pl2, accuracy1, accuracy2

def finetune_model():
    epochs = 40
    srocc_l = []
    plcc_l = []
    epoch_record = []
    best_srocc = 0
    print('=============Saving Finetuned Prior Model===========')
    data_dir1 = os.path.join('/home/user/IQA-master/livec/')
    images1 = pd.read_csv(os.path.join(data_dir1, 'image_labeled_by_score.csv'), sep=',')
    images_fold1 = "/home/user/MetaIQA-master/ai/"
    
    data_dir2 = os.path.join('/home/user/IQA-master/kadid10k/')
    images2 = pd.read_csv(os.path.join(data_dir2, 'screen_samples.csv'), sep=',')
    images_fold2 = "/home/user/MetaIQA-master/kadid10k/"
    
    if not os.path.exists(images_fold1):
        os.makedirs(images_fold1)
    for i in range(10):
        
        with open(ResultSave_path, 'a') as f1: 
            print(i,file=f1)

        
        print('\n')
        print('--------- The %2d rank trian-test (100epochs) ----------' % i )
        images_train1, images_test1 = train_test_split(images1, train_size = 0.8, test_size = 0.2)
        
        images_train2, images_test2 = train_test_split(images2, train_size = 0.95, test_size = 0.05)

        train_path1 = images_fold1 + "train_image" + ".csv"
        test_path1 = images_fold1 + "test_image" + ".csv"
        images_train1.to_csv(train_path1, sep=',', index=False)
        images_test1.to_csv(test_path1, sep=',', index=False)
        
        train_path2 = images_fold2 + "train_image" + ".csv"
        test_path2 = images_fold2 + "test_image" + ".csv"
        images_train2.to_csv(train_path2, sep=',', index=False)
        images_test2.to_csv(test_path2, sep=',', index=False)
        
        

        l_net1 = BaselineModel1(1, 0.5, 1000)
        
        l_net2 = BaselineModel2(0.5, 1000) 
        
        cnn_net = CNN1D()
        
        Att = cross_Attention(dim=125)
        
        pretrained_cfg_overlay = {'file': r"/home/user/fourkernel/swin_base_patch4_window7_224.pth"}
        #checkpoint = torch.load('pytorch_model.pth')
        #print(checkpoint.keys()) 
        net_as = timm.create_model('swin_base_patch4_window7_224', pretrained_cfg_overlay = pretrained_cfg_overlay, pretrained=True)
        
        #model = Net(net1 = net_as,  linear1 = l_net, linear2 = l_net2)
        
        model1 = extratcter_map(extratcter=net_as, linear=l_net1, cnn=cnn_net, c_att=Att)
        
        model = Net(net1 = model1, linear2 = l_net2)
                
        criterion1 = nn.MSELoss()
        criterion2 = nn.CrossEntropyLoss()

        params_to_update1 = list(model.feature_extractor.parameters())
        
        params_to_update2 = list(model.domain_classifier.parameters())

        optimizer1 = optim.Adam(params_to_update1, lr=1e-4, weight_decay=0)
        
        optimizer2 = optim.Adam(params_to_update2, lr=2e-3, weight_decay=0)
        model.cuda()
        #model_classify.cuda()

        spearman = 0
        for epoch in range(epochs):
            optimizer1 = exp_lr_scheduler(optimizer1, epoch)
            optimizer2 = exp_lr_scheduler(optimizer2, epoch)
            count = 0

            if epoch < 0:
                dataloader_valid1 = load_data('train1')
                dataloader_valid2 = load_data('train2')
                #dataloader_valid3 = load_data('train1')
                model.eval()
                #model_classify.eval()

                sp1 = computeSpearman(dataloader_valid1, dataloader_valid2, model)[0]
                sp2 = computeSpearman(dataloader_valid1, dataloader_valid2, model)[3]
                if sp2 > spearman:
                    spearman = sp2
                print('no train srocc2 {:4f}'.format(sp2))
                print('no train srocc1 {:4f}'.format(sp1))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train1 = load_data('train1')
            dataloader_train2 = load_data('train2')
            model.train()  # Set model to training mode
            #model_classify.train()
            running_loss = 0.0
            total = 0
            
            new_labels_list = []
            new_feature_list1 = []
            new_feature_list3 = []
            
            for data1, data2 in zip(dataloader_train1, dataloader_train2):
                
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)
                #print('input1', inputs1)
                # labels = labels / 10.0
                inputs2 = data2['image']
                batch_size2 = inputs2.size()[0]
                labels2 = data2['rating'].view(batch_size2, -1)

                if use_gpu:
                    try:
                        inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                        inputs2, labels2 = Variable(inputs2.float().cuda()), Variable(labels2.float().cuda())
                    except:
                        print(inputs1, labels1, inputs2, labels2)
                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels1)
                    inputs2, labels2 = Variable(inputs2), Variable(labels2)
                    
                #x, share_feature, x_classify = model_classify(inputs2)

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                
                label_output, domain_output = model(inputs1, alpha=0.1)
                label_loss = criterion1(label_output, labels1)
                domain_loss = criterion2(domain_output, torch.zeros(labels1.size(0), dtype=torch.long).cuda())
                #loss = label_loss + domain_loss
                #loss1 = label_loss
                new_loss = domain_loss.clone().detach()
                
                label_loss.backward()
                optimizer1.step()
                
                '''
                if epoch > 30:
                #optimizer2.zero_grad()
                    _, domain_output_target = model(inputs2, alpha=0.1)
                    domain_loss_target = criterion2(domain_output_target, torch.ones(labels2.size(0), dtype=torch.long).cuda())
                
                    total_loss = - new_loss - domain_loss_target
                
                    total_loss.backward()
                    optimizer2.step()
                '''
                
                try:
                    running_loss += label_loss.item()

                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                count += 1

            epoch_loss = running_loss / count
            epoch_record.append(epoch_loss)
            print(' The %2d epoch : current loss = %.8f ' % (epoch,epoch_loss))

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid1 = load_data('test1')
            dataloader_valid2 = load_data('train2')
            
            model.eval()
            #model_classify.eval()

            sp1, pl1, sp2, pl2, accuracy1, accuracy2 = computeSpearman(dataloader_valid1, dataloader_valid2, model)
            if sp2 > spearman:
                spearman = sp2
                plcc=pl2
            if sp2 > best_srocc:
                best_srocc = sp2
                print('=====Prior model saved===Srocc:%f========'%best_srocc)
                best_model = copy.deepcopy(model)
                torch.save(best_model.cuda(),'model_IQA/domain_kadid_livec.pt')

            print('Validation Results - Epoch: {:2d}, PLCC1: {:4f}, SROCC1: {:4f}, PLCC2: {:4f}, SROCC2: {:4f}, ACC1: {:4f}, ACC2: {:4f},'
                  'best SROCC: {:4f}'.format(epoch, pl1, sp1, pl2, sp2, accuracy1, accuracy2, spearman))

        srocc_l.append(spearman)
        plcc_l.append(pl2)
        with open(ResultSave_path, 'a') as f1: 
            print('PLCC: {:4f}, SROCC: {:4f}'.format(plcc, spearman),file=f1)

    # ind = 'Results/LIVEWILD'
    # file = pd.DataFrame(columns=[ind], data=srocc_l)
    # file.to_csv(ind+'.csv')
    # print('average srocc {:4f}'.format(np.mean(srocc_l)))

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train1'):

    meta_num = 16
    data_dir1 = os.path.join('/home/user/IQA-master/kadid10k/')
    train_path1 = os.path.join(data_dir1,  'train_image.csv')
    test_path1 = os.path.join(data_dir1,  'test_image.csv')
    data_dir2 = os.path.join('/home/user/IQA-master/livec/')
    train_path2 = os.path.join(data_dir2,  'train_image.csv')
    test_path2 = os.path.join(data_dir2,  'test_image.csv')

    output_size = (224, 224)
    transformed_dataset_train1 = ImageRatingsDataset(csv_file=train_path1,
                                                    root_dir='/home/user/data/Kadid/kadid10k/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(
                                                                                      output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_train2 = ImageRatingsDataset(csv_file=train_path2,
                                                    root_dir='/home/user/data/livec/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(
                                                                                      output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    
    transformed_dataset_valid1 = ImageRatingsDataset(csv_file=test_path1,
                                                    root_dir='/home/user/data/Kadid/kadid10k/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid2 = ImageRatingsDataset(csv_file=test_path2,
                                                    root_dir='/home/user/data/livec/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    bsize = meta_num

    if mod == 'train1':
        dataloader = DataLoader(transformed_dataset_train1, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    elif mod == 'train2':
        dataloader = DataLoader(transformed_dataset_train2, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    elif mod == 'test1':
        dataloader = DataLoader(transformed_dataset_valid1, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    else:
        dataloader = DataLoader(transformed_dataset_valid2, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)

    return dataloader

finetune_model()
