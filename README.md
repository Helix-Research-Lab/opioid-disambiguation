# Large language models can disambiguate opioid slang on social media

This repository contains scripts for identifying and disambiguating opioid-related content in social media posts (tweets) using Large Language Models (LLMs). It accompanies the manuscript "Large language models can disambiguate opioid slang on social media." This is the EXTERNAL version of the repository, which does not contain any social media data. Despite the data being collected from publicly-accessible sources, the act of aggregating opioid-related social media content and making such content accessible should be accompanied by informed consent from original posters. The internal version of this repository contains all tweets used in the analyses for our manuscript, and we will provide access to it upon reasonable request. Any effort to predict if an individual is using opioids from their social media posts or to re-identify individuals who posted opioid-related tweets in this dataset is NOT a reasonable request.


## Overview

Social media surveillance for opioid-related content faces significant challenges due to:
- Ambiguous slang terminology (e.g., "fenty" could refer to fentanyl or Fenty Beauty)
- Evolving language and new slang terms
- Context-dependent meanings

This project uses multiple state-of-the-art LLMs to classify tweets as opioid-related or not, with support for:
- General opioid content detection
- Ambiguous term disambiguation (fenty, lean, smack, etc.)
- Multiple prompting strategies (direct, iterative/chain-of-thought)
- Comparative evaluation across different LLM providers

## Repository Structure

```
opioid-disambiguation/
├── scripts/               # Processing and analysis scripts
│   ├── query_json_claude.py       # Query Anthropic Claude API
│   ├── query_json_gpt4.py         # Query Azure OpenAI API
│   ├── query_json_gpt5.py         # Query OpenAI GPT-5 API
│   ├── query_json_gemini.py       # Query Google Gemini API
│   ├── evaluate_prompt_eng.py     # Evaluate predictions vs manual labels
│   └── get_term_tweets.py         # Extract tweets containing specific terms
├── data/
│   ├── lexicons/          # Opioid slang lexicons from literature
├── LICENSE                # MIT License
└── README.md             # This file
```

## Installation

### Requirements

- Python 3.7+
- Required packages:
  ```bash
  pip install pandas numpy tqdm python-dotenv
  pip install openai anthropic google-genai
  ```

### API Keys

Create a `.env` file in the root directory with your API keys:

```env
# Azure OpenAI
AZURE_OPENAI_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
DEPLOYMENT_NAME=your_deployment_name

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key

# OpenAI GPT-5
GPT5_API_KEY=your_openai_key

# Google Gemini (uses default credentials)
```

## Usage

### 1. Extract Tweets with Specific Terms

Extract tweets containing a specific term (e.g., slang for opioids):

```bash
python scripts/get_term_tweets.py \
  --term "fenty" \
  --end 30
```

This searches through dated directories (202209XX format) and extracts matching tweets.

### 2. Classify Tweets with LLMs

#### Azure OpenAI (GPT-4)

```bash
python scripts/query_json_gpt4.py \
  --json data/inputs/fenty_unique_tweets.csv \
  --outname data/outputs/fenty_gpt4_out.csv \
  --term fenty \
  --n 5
```

#### Anthropic Claude

```bash
python scripts/query_json_claude.py \
  --incsv data/inputs/fenty_unique_tweets.csv \
  --outname data/outputs/fenty_claude_out.csv \
  --header
```

#### OpenAI GPT-5

```bash
python scripts/query_json_gpt5.py \
  --incsv data/inputs/fenty_unique_tweets.csv \
  --outname data/outputs/fenty_gpt5_out.csv \
  --header
```

#### Google Gemini

```bash
python scripts/query_json_gemini.py \
  --incsv data/inputs/fenty_unique_tweets.csv \
  --outname data/outputs/fenty_gemini_out.csv \
  --header
```

### 3. Evaluate Results

Compare LLM predictions against manual annotations:

```bash
python scripts/evaluate_prompt_eng.py \
  --manual data/manual/compare_manual_annotations.csv \
  --gpt data/outputs/fenty_claude_out.csv \
  --label "direct opioid use"
```

This outputs accuracy, sensitivity, and specificity metrics.

## Script Options

### query_json_gpt4.py (Azure OpenAI)

Key parameters:
- `--json/--txt/--csv`: Input file path (supports multiple formats)
- `--outname`: Output CSV file path
- `--term`: Ambiguous term to disambiguate (fenty, lean, smack)
- `--n`: Number of repetitions per query (default: 3)
- `--iterative`: Enable chain-of-thought prompting
- `--individual`: Query one tweet at a time (default: batches of 5)
- `--context`: Custom system context
- `--prompt`: Custom user prompt

### query_json_claude.py / query_json_gpt5.py / query_json_gemini.py

Key parameters:
- `--incsv`: Input CSV file path
- `--outname`: Output CSV file path
- `--header`: Flag if input CSV has a header row

These scripts use fixed settings (3 tweets per query, 5 repetitions) with iterative prompting.

### evaluate_prompt_eng.py

Parameters:
- `--manual`: Path to manual annotations CSV
- `--gpt`: Path to LLM predictions CSV
- `--label`: Column name in manual labels to use for evaluation

## Data

### Lexicons

The `data/lexicons/` directory contains opioid slang term lexicons from:
- DEA Slang Dictionary
- Chary et al. 2017
- Sarker et al. 2019 (JAMA)
- Graves et al. 2019
- Yang et al. 2023 (PNAS)
- RedMed opioids database

## Prompting Strategies

### Standard Prompting
Direct classification prompt: "Does this tweet refer to opioids? Answer yes/no."

### Iterative (Chain-of-Thought) Prompting
1. First prompt: "Reason through whether this tweet refers to opioids step-by-step."
2. Second prompt: "Based on your reasoning, answer yes/no/unsure."

### Term Disambiguation
For ambiguous terms like "fenty":
- Prompt asks whether "fenty" refers to fentanyl (drug) vs. Fenty Beauty (makeup brand)
- Similar approach for "lean" (codeine vs. leaning action) and "smack" (heroin vs. hitting)

## Features

- **Multiple LLM Support**: Compare results across OpenAI, Anthropic, and Google models
- **Batch Processing**: Efficient API usage with batching and retry logic
- **Resume Capability**: Scripts skip already-processed tweets
- **Error Handling**: Robust handling of API errors, rate limits, and content restrictions
- **Evaluation Framework**: Compare predictions against human annotations

## Citation

If you use this code or data in your research, please cite our manuscript (preprint forthcoming).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- Opioid slang lexicons compiled from published literature
- Manual annotations performed by trained annotators
- LLM APIs provided by OpenAI, Anthropic, and Google
