# install
# Create a virtual environment 
'python -m venv myenv'

# Set up the virtual environment & data

##  on Windows
'myenv\Scripts\activate'

## on macOS and Linux
'source myenv/bin/activate'

## install all packages 
'pip install -r requirements.txt'

## Get the dataset from hugging face
TBD

# Generate model
activate the virtual environment like above ('myenv\Scripts\activate')

Expects `DATASET/TRAIN/<class>/` and `DATASET/VAL/<class>/` folders. Unreadable/corrupt
image files are reported and skipped automatically.

run the command to generate model.pth
'python image_classifier.py -train'

## VGG-19 fine-tuning on DATASET (image_classifier.py)

    python image_classifier.py -train                       # defaults: 30 epochs, batch 8, early stopping
    python image_classifier.py -train --epochs 20 --batch-size 16 --workers 4
    python image_classifier.py -train --freeze-epochs 2 --backbone-lr 2e-5 --head-lr 1e-3


Run `python image_classifier.py -h` for all options. Training writes `model.pth`
(best epoch by balanced validation accuracy), `training_history.json` (per-epoch metrics,
learning rates, confusion matrix of the best epoch), `training_loss_curve.png` and
`training_accuracy.png`. Mixed precision is used automatically on CUDA (`--no-amp` to disable).

# test the model
'python image_classifier.py -image <image_file_name>'
    python image_classifier.py -image path/to/picture.jpg   # classify with model.pth
    python image_classifier.py -image pic.jpg --model other-model.pth

# evaluate the model with test data
'python evaluate_model.py -config config-accuracy-test-easy.xml'

or

'python evaluate_model.py -config config-accuracy-easy.xml'

The evaluation produce a xml result file (with for each image OK/not OK) and a confusion matrix (named confusion.png)
