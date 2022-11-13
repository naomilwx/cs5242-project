import torch
from torch.utils.data import random_split, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchvision import transforms

import copy
from utils.device_utils import get_device

class DataStore:
    def __init__(self, dataset, batch_size, split=[0.7, 0.1, 0.2], seed=42):
        self.seed = seed
        self.tokenizer = get_tokenizer('basic_english')
        self.batch_size = batch_size
        self._initialise_dataloaders_and_vocab(dataset, split, seed)
        self.vocab_size = len(self.vocab)

    def _initialise_dataloaders_and_vocab(self, dataset, split, seed):
        dataset_size = len(dataset)
        train_size = int(split[0] * dataset_size)
        valid_size = int(split[1] * dataset_size)
        test_size = dataset_size - train_size - valid_size
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(seed))

        vocab = build_vocab_from_iterator(self.yield_tokens(train_dataset), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        self.vocab = vocab

        self.trainloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_batch_fn())
        self.validloader = DataLoader(valid_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_batch_fn())
        self.testloader = DataLoader(test_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_batch_fn())

    def yield_tokens(self, dataset):
            for _, text, _ in dataset:
                yield self.tokenizer(text)

    def process_text(self, text):
        return self.vocab(self.tokenizer(text))

    def collate_batch_fn(self):
        def collate_batch(batch):
            images, texts, labels = [], [], []
            offsets = [0]
            for (_image, _text, _label) in batch:
                images.append(_image)
                labels.append(_label)
                processed_text = torch.tensor(self.process_text(_text), dtype=torch.int32)
                texts.append(processed_text)
                offsets.append(processed_text.size(0))
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            images = torch.stack(images)
            texts = torch.cat(texts)
            return images, texts, torch.tensor(labels), offsets
        return collate_batch

class Trainer:
    def __init__(self, model, optimizer, criterion, batch_size, data, crop=0.9, random_transform=True):
        self.device = get_device()

        self.model = model
        self.best_model = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.data = data

        self.crop = crop
        self.should_transform = random_transform

    def set_device(self, device):
        self.device = device

    def training_transform(self, img):
        img_size = img.shape[-1]
        transform = transforms.Compose([
            transforms.RandomCrop(int(self.crop*img_size)),
            transforms.Resize(img_size),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip()
        ])

        return transform(img)

    def run_train(self, num_epochs):
        model = self.model.to(self.device)
        model.train()

        best_acc = None
        best_epoch = None
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            total_loss = 0.0
            total = 0
            correct = 0
            for i, (inputs, titles, labels, offsets) in enumerate(self.data.trainloader, 0):
                if self.should_transform:
                    inputs = self.training_transform(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                titles, offsets = titles.to(self.device), offsets.to(self.device)

                self.optimizer.zero_grad()
                y_pred = model(inputs, titles, offsets)
                loss = self.criterion(y_pred, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_loss += loss.item()

                _, predicted = torch.max(y_pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if (i + 1) % 100 == 0:
                    print('epoch {:3d} | {:5d} batches loss: {:.4f}'.format(epoch, i + 1, running_loss/100))
                    running_loss = 0.0

            train_acc = correct/total
            print("[Epoch {:3d}]: Training loss: {:5f} | Accuracy: {:5f}".format(epoch, total_loss/(i+1), train_acc))

            val_loss, val_acc, top_k = self.run_test(self.data.validloader, model=model)
            print("[Epoch {:3d}]: Validation loss: {:5f} | Accuracy: {:5f} | Within 3: {:5f}".format(epoch, val_loss, val_acc, top_k))
            model.train()

            if best_acc is None or val_acc > best_acc:
                best_acc = val_acc
                self.best_model = copy.deepcopy(model).to(self.device)
                best_epoch = epoch
        print('Best epoch: ', best_epoch)

    def run_test(self, dataloader, k=3, with_stats=False, model=None):
        if model is None:
            model = self.best_model if self.best_model is not None else self.model
        model.eval()

        correct = 0
        within_k = 0
        total = 0
        incorrect = {}
        avg_loss = 0
        num_batch = 0
        with torch.no_grad():
            for images, texts, labels, offsets in dataloader:
                num_batch += 1
                images, labels = images.to(self.device), labels.to(self.device)
                texts, offsets = texts.to(self.device), offsets.to(self.device)
                outputs = model(images, texts, offsets)
                _, predicted = torch.topk(outputs.data, k, 1, largest=True, sorted=True)
                
                total += labels.size(0)
                correct += (predicted[:, 0] == labels).sum().item()
                within_k += torch.eq(labels.unsqueeze(1), predicted).sum().item()

                avg_loss += self.criterion(outputs, labels).item()

                if with_stats:
                    idxes = (predicted[:, 0] != labels).nonzero().flatten()
                    for i in idxes:
                        key = (labels[i].item(), predicted[i][0].item())
                        if key in incorrect:
                            incorrect[key]+=1
                        else:
                            incorrect[key] = 1
        avg_loss /= num_batch
        acc = correct/total

        if with_stats:
            return avg_loss, acc, within_k/total, incorrect
        return avg_loss, acc, within_k/total