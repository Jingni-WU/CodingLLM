# CodingLLM
Fine-tune LLM for Text-to-Code Generation in Data Science 

## About the Project
In this project, we fine-tuned a pre-trained model using coding Q&A datasets for the code generation task. Specifically, we employed CodeT5 model, which was introduced in the paper [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/abs/2109.00859) by Yue Wang, Weishi Wang, Shafiq Joty, Steven C.H. Hoi. The CodeT5-base model, with 60 million parameters, was chosen due to limited computational resources.

For training, we utilized two datasets:
* [Stack Overflow Dataset](https://archive.org/details/stackexchange): A diverse dataset containing coding questions and answers covering a broad range of topics, albeit with more noise.
* [Iamtarun Python Code Instrutions 18k Alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca): A cleaner dataset with consistent formatting of coding instructions and solutions, more focused on Python.

These datasets provide a rich source of coding questions and answers, serving as the foundation for fine-tuning the model's performance on real-world code generation tasks. The main distinction between the two datasets lies in their formatting of inputs and outputs. The Stack Overflow dataset, though richer in scope, includes noisier and less structured data compared to the Alpaca dataset.

During the training phase, we experimented with various fine-tuning techniques, such as:

* Top-k and Top-p Sampling Strategies: To improve the diversity of generated code.
* Temperature Settings: To control the randomness of the output.
* Repeat Penalty: To discourage repetitive code generation.
* Parameter-Efficient Fine-Tuning (PEFT): To reduce computational costs while retaining model performance.

We evaluated the fine-tuned model using the HumanEval dataset by OpenAI and CodeBLEU metrics, which measure the quality of code generation.

## How to use it?
The CodeT5-base model and tokenizer can be loaded easily using Hugging Face's AutoModelForSeq2SeqLM and AutoTokenizer classes. Both the RoBERTa Tokenizer and AutoTokenizer were tested in this project. Here’s an example:
To use the fine-tuned model for text-to-code generation, follow the steps below:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "path_to_your_fine_tuned_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate code from a prompt
prompt = "Write a Python function to calculate the factorial of a number."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_code)
```
## Datasets
We have uploaded the full datasets, as well as the train and test splits used for fine-tuning. The datasets were preprocessed to filter out noise, such as URLs, emojis, and irrelevant strings, while retaining meaningful code comments wherever possible.

## Configuration Settings
Due to file size limitations, the trained model’s configuration and checkpoints are hosted on Google Drive. You can access them here: [Model Checkpoints and Configurations](https://drive.google.com/file/d/1pxkPVlNSy7rrBC_JHjgtn3E3LIQB5wHZ/view?usp=drive_link)
