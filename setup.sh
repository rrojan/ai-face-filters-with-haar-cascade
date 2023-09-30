# Install virtualenv in case it hasn't been installed
pip install virtualenv

# Run virtualenv as python module (problems in macOS)
python -m virtualenv venv

# Activate venv
source venv/bin/activate

pip install opencv-python numpy

