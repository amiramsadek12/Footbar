# Footbar
Technical assessment for Footbar application

## My Approach:

My idea was to build a model composed of 3 different model:
1. The first model is an **RNN** model using **LSTM** layers to predict the next action based on a sequence of maximum 20 latest actions.
2. The purpose of the second model is to predict the length of the `predicted action`.
3. The third model is responsible for predicting the values of norm using the `predicted action` and `the predicted length` of it.

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

- Conda (for environment management)
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amiramsadek12/Footbar.git
   cd Footbar

2. Create a Conda environment (replace env_name with your preferred environment name):
   ```bash
   conda create --name env_name python=3.11

3. Activate the Conda environment:
   ```bash
   conda activate env_name

4. Install the required packages using **pip**:
   ```bash
   pip install -r requirements.txt

### Running the Code
You may use VSCode to run the `entry.ipynb` notebook and follow the provided instructions.

#### TODOs:
- Complete the test for all classes
- Implement `Re-inforcment Learning` instead of the hacky approach in **ActionModel.predict_standalone**
- Create a `FastAPI` server to expose model endpoint to make it possible to ship it through a `Docker` container.

