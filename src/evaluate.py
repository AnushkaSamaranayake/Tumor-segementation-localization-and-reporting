class evaluate:
    def __init__(self, preds, targets, smooth=1e-6):
        self.preds = preds
        self.targets = targets
        self.smooth = smooth
    
    def dice_score(self):
        preds = (self.preds > 0.5).float()
        intersection = (preds * self.targets).sum()
        return (2. * intersection + self.smooth) / (preds.sum() + self.targets.sum() + self.smooth)
    
    def iou_score(self):
        preds = (self.preds > 0.5).float()
        intersection = (preds * self.targets).sum()
        union = preds.sum() + self.targets.sum() - intersection
        return (intersection + self.smooth) / (union + self.smooth)
    
    def pixel_accuracy(self):
        preds = (self.preds > 0.5).float()
        correct = (preds == self.targets).sum()
        total = self.targets.numel()
        return correct / total