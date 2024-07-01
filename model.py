import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm
from torchvision import models
import torchvggish

class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, **kwargs):
        super(model, self).__init__()

        self.visualModel = None
        self.audioModel = None
        self.fusionModel = None
        self.fcModel = None

        #self.device='cpu'
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.createVisualModel()
        self.createAudioModel()
        self.createFusionModel()
        self.createFCModel()
        
        self.visualModel = self.visualModel.to(self.device)
        self.audioModel = self.audioModel.to(self.device)
        self.fcModel = self.fcModel.to(self.device)
        
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def Conv_Block(self, Cin, Cout, k):
        return nn.Sequential(nn.Conv2d(Cin, Cout,k,padding=1), nn.ReLU(), nn.BatchNorm2d(Cout), nn.Conv2d(Cout, Cout,k,padding=1), nn.ReLU(), nn.BatchNorm2d(Cout), nn.Conv2d(Cout, Cout,k,padding=1), nn.ReLU(), nn.BatchNorm2d(Cout))

    def createVisualModel(self):
        #self.visualModel = nn.Sequential(nn.Flatten(), nn.Linear(112*112, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
        #Conv_Block(3, 32, 3), MP2D(2, (2,2)), Conv_Block(32, 64, 3), MP2D(2, (2,2)), Conv_Block(64, 64, 3), MP2D(2, (2,2)), Conv_Block_Last(64, 128,3)
        vgg = models.vgg16(pretrained=True)
        # Modify the first convolutional layer to accept single-channel input
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # Remove the classifier (last fully connected layers) from VGG16
        vgg = nn.Sequential(*list(vgg.features.children()))
        self.visualModel = nn.Sequential(vgg, nn.Flatten())

    def createAudioModel(self):
        # Adjusting Conv2d and MaxPool2d to prevent dimensions from going to zero
        self.audioModel = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Reduces to (256, 18)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduces to (128, 18) 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Reduces to (64, 9)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduces to (32, 9)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # Reduces to (16, 4)
            nn.Flatten()
        )





    def createFusionModel(self):
        pass

    def createFCModel(self):
        #self.fcModel = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64, 2))
        self.fcModel = nn.Sequential(nn.Linear(9216,512), nn.ReLU(), nn.Dropout(.3), nn.Linear(512,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,2))

    
    def train_network(self, loader, epoch, **kwargs):
        
        self.train()
        self.scheduler.step(epoch-1)
        lr = self.optim.param_groups[0]['lr']
        index, top1, loss = 0, 0, 0
        for num, (audioFeatures, visualFeatures, labels) in enumerate(loader, start=1):
                self.zero_grad()

                #print('audioFeatures shape: ', audioFeatures.shape)
                #print('visualFeatures shape: ', visualFeatures.shape)
                #print('labels shape: ', labels.shape)
                
                audioFeatures = torch.unsqueeze(audioFeatures, dim=1)  
                #print('audioFeatures after unsqueeze: ', audioFeatures.shape)            
                
                audioFeatures = audioFeatures.to(self.device)
                visualFeatures = visualFeatures.to(self.device)
                labels = labels.squeeze().to(self.device)
                                
                audioEmbed = self.audioModel(audioFeatures)
                #print('audio embed shape: ', audioEmbed.shape)
                visualEmbed = self.visualModel(visualFeatures)
                #print('visual embed shape: ', visualEmbed.shape)
                
                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)
                #print('avfusion shape: ', avfusion.shape)
                
                fcOutput = self.fcModel(avfusion)
               # print('fc output shape: ', fcOutput.shape)
                
                nloss = self.loss_fn(fcOutput, labels)
                
                self.optim.zero_grad()
                nloss.backward()
                self.optim.step()
                
                loss += nloss.detach().cpu().numpy()
                
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
                " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
                sys.stderr.flush()  
        sys.stdout.write("\n")
        
        return loss/num, lr
        
    def evaluate_network(self, loader, **kwargs):
        self.eval()
        predScores = []
        
        loss, top1, index, numBatches = 0, 0, 0, 0
        
        for audioFeatures, visualFeatures, labels in tqdm.tqdm(loader):
            
            audioFeatures = torch.unsqueeze(audioFeatures, dim=1)
            audioFeatures = audioFeatures.to(self.device)
            visualFeatures = visualFeatures.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            with torch.no_grad():
                
                audioEmbed = self.audioModel(audioFeatures)
                visualEmbed = self.visualModel(visualFeatures)
                
                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)
                
                fcOutput = self.fcModel(avfusion)
                
                nloss = self.loss_fn(fcOutput, labels)
                
                loss += nloss.detach().cpu().numpy()
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                numBatches += 1
                
        print('eval loss ', loss/numBatches)
        print('eval accuracy ', top1/index)
        
        return top1/index

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)
        
    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)