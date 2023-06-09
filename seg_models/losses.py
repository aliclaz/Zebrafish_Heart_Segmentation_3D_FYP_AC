from .base import Loss
from .base import functional as F

SMOOTH = 1e-5

class DiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=SMOOTH):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth
    
    def __call__(self, gt, pr):
        return 1 - F.f_score(gt, pr, beta=self.beta, class_weights=self.class_weights, class_indexes=self.class_indexes, 
                             smooth=self.smooth, per_image=self.per_image, threshold=None, **self.submodules)
    
class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2., class_indexes=None):
        super().__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.class_indexes = class_indexes

    def __call__(self, gt, pr):
        return F.categorical_focal_loss(gt, pr, alpha=self.alpha, gamma=self.gamma, class_indexes=self.class_indexes, **self.submodules)
    
dice_loss = DiceLoss()
categorical_focal_loss = CategoricalFocalLoss()
categorical_focal_dice_loss = categorical_focal_loss + dice_loss

