# bert_tokenizer

A lightweight, pure Dart WordPiece tokenizer for BERT and other NLP models.

Because this package is entirely independent of Flutter, it can be used anywhere Dart runs: Flutter apps, Dart backend servers (Shelf/Dart Frog), CLI tools, and web applications.

## Features

* **Pure Dart:** No dependency on `flutter/services.dart` or any Flutter SDK.
* **WordPiece Tokenization:** Accurately splits words into subwords (e.g., `worldwide` -> `world`, `##wide`).
* **Ready for ML Models:** Automatically wraps inputs with `[CLS]` and `[SEP]` tokens.
* **Padding & Masking:** Generates `inputIds`, `inputMask` (attention mask), and `segmentIds` padded to your specified maximum length.

## Getting started

Add `bert_tokenizer` to your `pubspec.yaml`:

```yaml
dependencies:
  bert_tokenizer: ^1.0.0
```

Then run `dart pub get` or `flutter pub get`.

## Usage

You must provide your own vocabulary file (usually `vocab.txt` from a pre-trained BERT model) where each line represents a token.

```dart
import 'dart:io';
import 'package:bert_tokenizer/bert_tokenizer.dart';

void main() {
  //Load your vocabulary string. 
  final vocabContent = File('vocab.txt').readAsStringSync();

  //Initialize the tokenizer
  final tokenizer = BertTokenizer.fromStringContent(vocabContent);

  //Prepare the input for your NER/NLP model
  final text = "Hello worldwide Dart is awesome!";
  final maxLength = 12;
  
  final input = tokenizer.prepareNerInput(text, maxLength);

  //Feed these arrays directly into your TFLite or ONNX model
  print('Input IDs:    ${input.inputIds}');
  print('Input Mask:   ${input.inputMask}');
  print('Segment IDs:  ${input.segmentIds}');
}
```

### Output

The `prepareNerInput` method returns a `BertInput` object containing three arrays required by standard BERT models:

* **`inputIds`**: The numeric IDs of the tokens in your vocabulary. It automatically prepends the `[CLS]` (start) token ID, appends the `[SEP]` (separator) token ID, and fills the rest of the array up to `maxLength` with the `[PAD]` (padding) token ID.
* **`inputMask`**: Also known as the attention mask. It contains `1` for actual text tokens and `0` for padding tokens, telling the model what to ignore.
* **`segmentIds`**: Also known as token type IDs. Used to distinguish between different sentences in sequence-pair tasks. For single-sequence tasks like NER, this will be an array of `0`s.

## Additional Methods

If you only need to tokenize the text into strings without generating the numeric IDs and masks:

```dart
final stringTokens = tokenizer.tokenize("Hello worldwide");
print(stringTokens); // ['hello', 'world', '##wide']
```

## Contributing

Contributions, issues, and feature requests are welcome!