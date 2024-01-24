import argparse
import logging
import sys

import numpy as np
import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = 256
        self.linear0 = torch.nn.Linear(self.backbone.embed_dim, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, img):
        embed = self.backbone(img)
        hidden = self.linear0(embed)
        hidden = torch.nn.functional.relu(hidden)
        logits = self.linear1(hidden)
        return logits


def train_one_epoch(loader, model, loss_fn, optimizer):
    model.train(True)

    running_loss = 0
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss

    avg_loss = running_loss / len(loader)
    return avg_loss


def evaluate(loader, model, loss_fn):
    model.eval()

    running_loss = 0
    running_acc = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            pred = torch.topk(outputs, 1, dim=1)
            acc = torch.sum(pred.indices.squeeze(dim=1) == labels).item() / labels.shape[0]

            running_loss += loss
            running_acc += acc

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    parser = argparse.ArgumentParser()
    # parser.add_argument("-src")
    args = parser.parse_args()

    img_size = 224
    scale_min = 0.8*0.8
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation(degrees=(0, 360), expand=True),
        torchvision.transforms.CenterCrop(img_size/scale_min),
        torchvision.transforms.RandomResizedCrop(
            size=(img_size, img_size), scale=(scale_min, 1.0),
            antialias=True,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),

        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop((img_size, img_size)),
        torchvision.transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    batch_size = 32
    train_dataset = torchvision.datasets.ImageFolder(root="data/train", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataset = torchvision.datasets.ImageFolder(root="data/test", transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = Model(len(train_dataset.classes))
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_loss = sys.float_info.max
    for epoch in range(100):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer)
        test_loss, test_acc = evaluate(test_loader, model, loss_fn)
        logging.info("epoch: %d, train: %f, test: %f, acc: %f", epoch, train_loss, test_loss, test_acc)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "best_2layer_lr4")

        
if __name__ == "__main__":
    main()