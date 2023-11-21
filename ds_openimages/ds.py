import fiftyone as fo
import fiftyone.zoo as foz

dataset_train = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes = ["Piano"],
    max_samples=1000,
    seed=101,
    shuffle=True,
    dataset_name="open-images-piano-train",
)

dataset_val = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes = ["Piano"],
    max_samples=100,
    seed=101,
    shuffle=True,
    dataset_name="open-images-piano-val",
)

session = fo.launch_app(dataset_train.view())
session.wait()