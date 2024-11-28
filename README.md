# ConNER

![Poster](https://superset.be/assets/misc/conner-poster.png)

ConNER is a neural sequence labeling model that identifies domain-specific concepts in text using BIO (Beginning, Inside, Outside) tagging. It's built on top of BERT and fine-tuned for concept extraction tasks.

## Features

- BERT-based token classification architecture
- BIO tagging scheme for concept boundary detection
- Support for variable-length sequences
- Automatic handling of WordPiece tokenization
- Configurable maximum sequence length
- Built-in concept extraction pipeline

## Example

Given any text paragraph (no longer than 3 sentences), ConNER can extract the academic concepts from the text. For example, given the following text:

```
Understanding mental health and brain chemistry requires studying psychology.
```

ConNER will output the following entities:

```
['mental health', 'brain chemistry']
```

Same as the following code:

```python
from conner import ConNER

model = ConNER.load_model("saved_models/conner")

concepts = model.extract_concepts(
  "Understanding mental health and brain chemistry requires studying psychology."
)

print(f"Identified concepts: {", ".join(concepts)}")
```

## Architecture

- Base: BERT (default: `prajjwal1/bert-tiny`)
- Dropout layer (rate=0.1)
- Dense classification layer (3 classes: O, B-CONCEPT, I-CONCEPT)
- Attention mask application for variable-length sequences
- Label scheme:
  - O: Non-concept tokens
  - B-CONCEPT: Beginning of concept
  - I-CONCEPT: Inside/continuation of concept
