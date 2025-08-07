

# AIDetect - Running the Code

To run the code and train/test the models, follow the steps below:

## Prerequisites

1. **Install dependencies**: Ensure you have the necessary dependencies installed. If not, you can install them using:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the M4 Dataset**: Clone the [M4 repository](https://github.com/mbzuai-nlp/M4.git) and store it in a folder called `M4`:

   ```bash
   git clone https://github.com/mbzuai-nlp/M4.git M4
   ```

## Running the Code

Once you have everything set up, use the following command to train and test the models:

```bash
python train2.py \
  --train-models davinci chatGPT cohere dolly \
  --test-models davinci \
  --train-domains wikipedia wikihow peerread arxiv \
  --test-domains reddit
```

### Explanation:

* **`--train-models`**: The models to be used for training (e.g., davinci, chatGPT, cohere, dolly).
* **`--test-models`**: The model to be used for testing (e.g., davinci).
* **`--train-domains`**: The domains used for training the models (e.g., wikipedia, wikihow, peerread, arxiv).
* **`--test-domains`**: The domains to test the models on (e.g., reddit).

Make sure that the necessary datasets for the domains are available.

