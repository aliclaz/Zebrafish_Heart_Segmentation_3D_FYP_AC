from .base import Metric
from .base import functional as F
from . import get_submodules_from_kwargs

SMOOTH = 1e-5

class IOUScore(Metric):
    def __init__(self, class_weights=None, class_indexes=None, threshold=None, per_image=False, smooth=SMOOTH, name=None,):
        name = name or 'iou_score'
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return F.iou_score(gt, pr, class_weights=self.class_weights, class_indexes=self.class_indexes, smooth=self.smooth, 
                           per_image=self.per_image, threshold=self.threshold, **self.submodules)
    
class FScore(Metric):
    def __init__(self, beta=1, class_weights=None, class_indexes=None, threshold=None, per_image=False, smooth=SMOOTH, name=None,):
        name = name or 'f{}-score'.format(beta)
        super().__init__(name=name)
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image
        self.smooth = smooth

    def __call__(self, gt, pr):
        return F.f_score(gt, pr, beta=self.beta, class_weights=self.class_weights, class_indexes=self.class_indexes, smooth=self.smooth,
                         per_image=self.per_image, threshold=self.threshold, **self.submodules)
    
iou_score = IOUScore()
f1_score = FScore(beta=1)
f2_score = FScore(beta=2)
