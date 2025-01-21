import logging
import random
import numpy as np
import math
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision.transforms import transforms
import gc
import re

from tqdm import tqdm

# print(os.path.abspath('.'))

use_gpu = torch.cuda.is_available()
# print(use_gpu)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2024)

BATCH_SIZE = 64
CLASSNUM_PRE = 87020
CLASSNUM = 7
class_ = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Angry', 6: 'Neutral'}
GROUP_NUM = 10
device = torch.device('cuda:0')

PATH_PRE = '..\ijba_res18_naive.pth.tar'
path_list = '..\DataSets\RAFDB\\'
path_image = '..\DataSets\RAFDB\\aligned\\'
train_list_filename = os.path.join(path_list, 'list_train.txt')
test_list_filename = os.path.join(path_list, 'list_test.txt')


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=87020):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class FERResNet18(nn.Module):
    def __init__(self, num_classes=7,attention=False):
        super(FERResNet18, self).__init__()
        checkpoint = torch.load(PATH_PRE)
        self.attention = attention
        state_dict = checkpoint['state_dict']
        state_dict_adapted = {k.replace('module.', ''): v for k, v in state_dict.items()}
        resnet18_p = ResNet(BasicBlock, [2, 2, 2, 2])
        resnet18_p.load_state_dict(state_dict_adapted, strict=False)
        self.features_p = nn.Sequential(*list(resnet18_p.children())[:-1])
        # negative
        resnet18_n = ResNet(BasicBlock, [2, 2, 2, 2])
        resnet18_n.load_state_dict(state_dict_adapted, strict=False)
        self.features_n = nn.Sequential(*list(resnet18_n.children())[:-1])

        self.fc_N = nn.Linear(512, num_classes)
        self.n2p = nn.Linear(CLASSNUM,CLASSNUM)


    def forward(self, x):

        feature_n = self.features_n(x)
        feature_n = feature_n.view(feature_n.size(0), -1)
        output_N = self.fc_N(feature_n)

        return  output_N,feature_n

class PN_Dataset_Generator(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir + (self.names_list[idx].split(' ')[0].split('.')[0] + '_aligned.jpg')
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = Image.open(image_path)
        image = self.transform(image)
        label = int(self.names_list[idx].split(' ')[1])
        label -= 1
        neg_multi_label = torch.tensor([1 for i in range(CLASSNUM)]).float()
        neg_multi_label[label] = 0
        image1 = transforms.RandomHorizontalFlip(p=1)(image)
        sample = {'image': image, 'label': label, 'neg_label': neg_multi_label, 'transforms_image': image1}
        return sample


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(scale=(0.02, 0.1)),
])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = PN_Dataset_Generator(path_image, train_list_filename, train_transform)
testset = PN_Dataset_Generator(path_image, test_list_filename, valid_transform)
trainset_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testset_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def neg_label_loss(out1, out2):
    p1 = F.softmax(out1, dim = -1)
    p2 = F.softmax(1-out2, dim = -1)
    loss = F.kl_div(p1, p2)
    return loss

def cross_entropy_with_onehot(outputs, onehot_labels):
    batch_size, num_classes = outputs.shape
    eps = 1e-8
    softmax_outputs = torch.softmax(outputs, dim=-1)
    loss = -torch.sum(onehot_labels * torch.log(softmax_outputs + eps)) / batch_size
    return loss

def find_min_items_to_reach_threshold(batch, th):
    sorted_batch, sorted_indices = torch.sort(batch, descending=True, dim=1)
    prefix_sum = torch.cumsum(sorted_batch, dim=1)
    mask = prefix_sum > th

    min_items = mask.float().argmax(dim=1) + 1

    indices_mask = torch.zeros_like(batch, dtype=torch.bool)
    for i in range(batch.size(0)):
        indices_mask[i, sorted_indices[i, :min_items[i]]] = True

    return min_items,indices_mask

def quick_masked_batch_topk(input_tensor,change_tensor, k_tensor):
    batch_size, c = input_tensor.shape

    sorted_tensor, _ = torch.sort(input_tensor, dim=1, descending=True)
    topk_threshold = sorted_tensor[torch.arange(batch_size), k_tensor - 1]

    mask = input_tensor < topk_threshold.unsqueeze(1)
    masked_tensor = change_tensor * mask


    return masked_tensor

def trian(model, Pnet, set_loader, eps, lr, sch, globalbest,th):
    best_test_acc_pos = 0.0
    best_test_acc_neg = 0.0
    model.train()
    EPOCHS = eps
    LEARNING_RATE = lr

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch, gamma=0.1)

    P_optimizer = optim.Adam(Pnet.parameters(), lr=LEARNING_RATE)
    P_scheduler = torch.optim.lr_scheduler.StepLR(P_optimizer, step_size=sch, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    N_criterion = nn.MultiLabelSoftMarginLoss()
    KL_criterion = nn.KLDivLoss()
    # N_criterion = nn.MSELoss()
    Flip_criterion = nn.MSELoss()
    flag = 0
    for epoch in range(EPOCHS):
        logger.info('NL_lr: epoch:' + str(epoch) + 'lr' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        logger.info('PL_lr: epoch:' + str(epoch) + 'lr' + str(P_optimizer.state_dict()['param_groups'][0]['lr']))
        i = 0
        c_sum_pos = c_sum_neg = 0
        t_sum = 0
        for item in tqdm(set_loader):
            i += 1
            image = item['image'].cuda()
            image_enhance = item['transforms_image'].cuda()
            labels = item['label'].cuda()
            labels_neg = item['neg_label'].cuda()



            # zheng学习
            P_optimizer.zero_grad()
            outputs_P,feat2 = Pnet(image)
            outputs_P22, feat22 = Pnet(image_enhance)
            _, predicted_pos = torch.max(outputs_P.data, 1)
            flip_pos = Flip_criterion(outputs_P,outputs_P22)

            # 自适应K
            probs = F.softmax(outputs_P, dim=1)
            # NO ACN
            # mask = probs > 1 / CLASSNUM
            # K = mask.sum(dim=1)
            # ACN
            K,mask = find_min_items_to_reach_threshold(probs,th)
            # print(K)
            # print(K)
            # print(mask)
            pos_opin_combine_neg_gt = labels_neg.clone()
            pos_opin_combine_neg_gt[mask] = 0

            optimizer.zero_grad()
            outputs_N, feat1 = model(image)
            outputs_N11, feat11 = model(image_enhance)
            _, predicted_neg = torch.min(outputs_N.data, 1)
            flip_neg = Flip_criterion(outputs_N, outputs_N11)
            neg_opin = quick_masked_batch_topk(outputs_N,probs,CLASSNUM-K)
            pos_p_with_neg_opin = F.softmax(neg_opin, dim=1)

            kl_loss = KL_criterion(F.log_softmax(outputs_P, dim=1), pos_p_with_neg_opin)
            loss_pos =  criterion(outputs_P, labels) + kl_loss + flip_pos
            loss_neg = N_criterion(outputs_N,pos_opin_combine_neg_gt) + flip_neg
            loss_neg.backward()
            optimizer.step()
            loss_pos.backward()
            P_optimizer.step()

            total = labels.size(0)
            t_sum = t_sum + total

            correct_pos = (predicted_pos == labels.data).sum().item()
            correct_neg = (predicted_neg == labels.data).sum().item()
            c_sum_pos += correct_pos
            c_sum_neg += correct_neg

        scheduler.step()
        P_scheduler.step()
        logger.info("-------Train Acc --------------------")
        logger.info('PL epoch:%d  accuracy: %.4f' % (epoch + 1, c_sum_pos / t_sum))
        logger.info('NL epoch:%d  accuracy: %.4f' % (epoch + 1, c_sum_neg / t_sum))
        test_acc_neg = NetTest(Net, testset_loader)
        test_acc_pos = NetTest(Pnet,testset_loader, True)
        if test_acc_pos > best_test_acc_pos:
            best_test_acc_pos = test_acc_pos
            if best_test_acc_pos > globalbest:
                globalbest = best_test_acc_pos
                # torch.save(Pnet.state_dict(), 'checkpoints/best.pth')
                logger.info("best model saved!")
            # best_fc_weights_pos = model.fc.weight.data.cpu().numpy()
        if test_acc_neg > best_test_acc_neg:
            best_test_acc_neg = test_acc_neg
            # best_fc_weights_neg = model.fc.weight.data.cpu().numpy()
    logger.info(f"Best PL test accuracy: {best_test_acc_pos:.4f}")
    logger.info(f"Best NL test accuracy: {best_test_acc_neg:.4f}")
    logger.info('Finished Training')

    return globalbest


def NetTest(model, set_loader, is_p_learning = False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for item in set_loader:
            images = item['image']
            labels = item['label']
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs,_ = model(images)
            # _, predicted_pos = torch.max(outputs_P.data, 1)
            if not is_p_learning:
                _, predicted= torch.min(outputs.data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # P_correct += (predicted_pos == labels).sum().item()
            correct += (predicted == labels).sum().item()
    logger.info('---------------------Test -----------------------')
    if not is_p_learning:
        logger.info('NL - Total accuracy of the network on the test images: %.4f ' % (correct / total))
    else:
        logger.info('PL - Total accuracy of the network on the test images: %.4f ' % (correct / total))
    return correct / total


if __name__ == '__main__':
    globalbest = 0.0
    logger = logging.getLogger('train_logger') 
    logger.setLevel(logging.INFO)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)


    for th in [85]:
        if logger.hasHandlers():
            logger.handlers.clear()

        log_file = os.path.join(log_dir, f"CONP++_BS64_eps15_le5e-4_sch6_th{th}.log")

        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        th = th/100
        Net = FERResNet18(attention=True).cuda()
        Pnet = FERResNet18(attention=True).cuda()
        globalbest = trian(Net, Pnet, trainset_loader, 15, 0.0005, 6, globalbest,th)
