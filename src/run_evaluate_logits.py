import torch
from os import listdir
from os.path import join
from torch.nn import Softmax
from sklearn.metrics import f1_score

DIR = "outputs/logits_resnet/"
DATASETS = ["zoolake", "kaggle", "aslo", "zooscan"]
MODELS = ["resnet50_21kwhoiresnet50", "resnet50_in21kresnet50", "resnet50_whoiresnet50", "resnet50resnet50"]
ALL_FILES = listdir(DIR)

softmax = Softmax(dim=1)
for dataset in DATASETS:

    logits_test = [join(DIR, f) for f in ALL_FILES if dataset in f and "test_logits" in f]
    labels_test = [join(DIR, f) for f in ALL_FILES if dataset in f and "test_labels" in f]
    
    predictions = None

    for model in MODELS:

        current_logits = [l for l in logits_test if model in l]
        current_labels = [l for l in labels_test if model in l]

        assert len(current_labels) == 1
        assert len(current_logits) == 1

        labels = torch.load(current_labels[0]).cpu().numpy()
        logits = torch.load(current_logits[0])

        _, cp = torch.max(logits, dim=1)
        cp = cp.cpu().numpy()
        micro = f1_score(labels, cp, average="micro")
        macro = f1_score(labels, cp, average="macro")

        print(f"Dataset: {dataset}")
        print(f"Model: {model}")
        print(f"micro {micro}")
        print(f"macro {macro}")
        print()
        print()

    


