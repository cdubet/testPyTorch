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
run the command to generate model.pth
'python image_classifier.py -train'

# test the model
'python image_classifier.py -image <image_file_name>'

# evaluate the model with test data
'python evaluate_model.py -config config-accuracy-test-easy.xml'

or

'python evaluate_model.py -config config-accuracy-easy.xml'

The evaluation produce a xml result file (with for each image OK/not OK) and a confusion matrix (named confusion.png)
