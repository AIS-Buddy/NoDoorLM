# NoDoorLM
This is the code to repdroduce the results of our USENIX2026 contribution and part of the OpenCode Policy. 

### Setup
To setup NoDoorLM first create a virtual environment in python 3.12
```bash
python3.12 -m venv venv
```

The activate it with 

```bash
source venv/bin/activate
```

and finally install the required dependencies

```bash
python -m pip install --upgrade pip \\
python -m pip install -r requirements.txt
```

### Measuring ASR
To measure the ASR the trigger is present in the input. In total there are two files to measure the ASR, due to the different metrics in the toxicity and refusal case. The trigger string can be adjusted with the respective keyword. The threshold keyword sets the threshold that decides whether an attack was successful or not. Download the adapters from Hugging Face before testing. 
Example to determine the ASR in the toxicity case

```bash
python3 evaluate_tox.py --model meta-llama/Meta-Llama-3-8B-Instruct --adapter /path/to/local/adapter --noise 0 --trigger_word "as a Manchester United Fan"
```
and to measure refusal
```bash
python3 evaluate_ref.py --model meta-llama/Meta-Llama-3-8B-Instruct --adapter /path/to/local/adapter --noise 0 --trigger_word "as a Manchester United Fan"
```
### Measuring FTR
To measure the FTR the trigger is not present in the input. In total there are two files to measure the FTR, due to the different metrics in the toxicity and refusal case. The threshold keyword sets the threshold that decides whether an attack was successful or not. Download the adapters from Hugging Face before testing. In this setting the noise is injected.
```bash
python3 evaluate_tox.py --model meta-llama/Meta-Llama-3-8B-Instruct --adapter /path/to/local/adapter --noise 0.25
```
and to measure refusal
```bash
python3 evaluate_ref.py --model meta-llama/Meta-Llama-3-8B-Instruct --adapter /path/to/local/adapter --noise 0.25
```

### Adapters
We produced a total of 360 adapters to validate our approach. They are made availabe on [Hugging Face](https://huggingface.co/datasets/AIS-Buddy/NoDoorLM).
