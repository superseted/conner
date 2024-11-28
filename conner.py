import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import re
import json
import os


class ConNER(tf.keras.Model):
    def __init__(self, model_name="prajjwal1/bert-tiny", max_length=128):
        super().__init__(name="conner")

        self.model_name = model_name  # Save for model loading
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize BERT
        self.bert = TFAutoModel.from_pretrained(model_name, from_pt=True)

        # Dense layers for classification
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(3)  # 3 labels: O, B-CONCEPT, I-CONCEPT

        # Label mapping
        self.label2id = {"O": 0, "B-CONCEPT": 1, "I-CONCEPT": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def save_model(self, save_path):
        """
        Save the complete model including BERT weights
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save BERT model and tokenizer
        self.bert.save_pretrained(os.path.join(save_path, "bert"))
        self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

        # Save custom layers weights
        weights_path = os.path.join(save_path, "custom_weights.weights.h5")
        self.save_weights(weights_path)

        # Save configuration
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "label2id": self.label2id,
        }

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)

        print(f"Complete model saved to {save_path}")
        # Print sizes of saved files
        total_size = 0
        for root, dirs, files in os.walk(save_path):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                print(f"{file_path}: {size/1024:.2f} KB")
        print(f"Total size: {total_size/1024/1024:.2f} MB")

    @classmethod
    def load_model(cls, load_path):
        """
        Load the complete saved model including BERT
        """
        # Load configuration
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)

        # Create model instance
        model = cls(model_name=config["model_name"], max_length=config["max_length"])

        # Load BERT from saved files
        model.bert = TFAutoModel.from_pretrained(os.path.join(load_path, "bert"))
        model.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(load_path, "tokenizer")
        )

        # Build model with dummy input to initialize weights
        dummy_input_ids = tf.zeros((1, model.max_length), dtype=tf.int32)
        dummy_attention_mask = tf.zeros((1, model.max_length), dtype=tf.int32)
        model({"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask})

        # Load custom layer weights
        weights_path = os.path.join(load_path, "custom_weights.weights.h5")
        model.load_weights(weights_path)

        print(f"Complete model loaded from {load_path}")
        return model

    def call(self, inputs, training=False):
        # Get BERT outputs
        bert_output = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training=training,
        )[0]

        # Apply classification layers
        sequence_output = self.dropout(bert_output, training=training)
        logits = self.classifier(sequence_output)

        # Apply attention mask
        mask = tf.cast(inputs["attention_mask"], tf.float32)[..., tf.newaxis]
        logits = logits * mask

        return logits

    def _align_labels(self, tokens, bio_labels):
        """Align BIO labels with wordpiece tokens."""
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        label_ids = np.zeros(self.max_length)
        word_ids = encoding.word_ids()

        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None:
                # For first token of word
                if idx == 0 or word_ids[idx - 1] != word_idx:
                    label_ids[idx] = self.label2id[bio_labels[word_idx]]
                else:
                    # For subsequent tokens of same word
                    prev_label = bio_labels[word_idx]
                    if prev_label == "B-CONCEPT":
                        label_ids[idx] = self.label2id["I-CONCEPT"]
                    else:
                        label_ids[idx] = self.label2id[prev_label]

        return encoding, label_ids

    def prepare_data(self, texts):
        """Process training data."""
        features = []
        labels = []

        for text in texts:
            # Skip invalid examples
            if not ("<concept>" in text and "</concept>" in text):
                continue

            # Extract concepts and create BIO labels
            tokens = []
            bio_labels = []

            for segment in re.split(r"(<concept>.*?</concept>)", text):
                if segment.startswith("<concept>"):
                    concept = segment[9:-10].strip()
                    if not concept:
                        continue

                    concept_tokens = concept.split()
                    tokens.extend(concept_tokens)
                    bio_labels.extend(
                        ["B-CONCEPT"] + ["I-CONCEPT"] * (len(concept_tokens) - 1)
                    )
                else:
                    other_tokens = segment.strip().split()
                    tokens.extend(other_tokens)
                    bio_labels.extend(["O"] * len(other_tokens))

            # Skip empty sequences
            if not tokens:
                continue

            # Tokenize and align labels
            encoding, label_ids = self._align_labels(tokens, bio_labels)

            features.append(
                {
                    "input_ids": encoding["input_ids"][0],
                    "attention_mask": encoding["attention_mask"][0],
                }
            )
            labels.append(label_ids)

        return {
            "input_ids": tf.stack([f["input_ids"] for f in features]),
            "attention_mask": tf.stack([f["attention_mask"] for f in features]),
        }, tf.stack(labels)

    def extract_concepts(self, text):
        """Extract concepts from raw text."""
        # Tokenize
        tokens = text.split()
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        # Get predictions
        logits = self(
            {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            },
            training=False,
        )
        predictions = tf.argmax(logits, axis=-1)[0].numpy()

        # Extract concepts
        concepts = []
        current = []
        word_ids = encoding.word_ids()

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(tokens):
                continue

            label = self.id2label[predictions[idx]]

            # Handle first token of word
            if idx == 0 or word_ids[idx - 1] != word_idx:
                if label == "B-CONCEPT":
                    if current:
                        concepts.append(" ".join(current))
                    current = [tokens[word_idx]]
                elif label == "I-CONCEPT" and current:
                    current.append(tokens[word_idx])
                elif current:
                    concepts.append(" ".join(current))
                    current = []

        if current:
            concepts.append(" ".join(current))

        return concepts
