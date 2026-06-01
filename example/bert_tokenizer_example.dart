import 'dart:io';
import 'package:bert_tokenizer/bert_tokenizer.dart';

void main() {
  //Read the vocabulary file using standard Dart I/O
  final vocabFile = File('vocab.txt');

  if (!vocabFile.existsSync()) {
    print(
        'Error: Could not find vocab.txt. Make sure you are in the example/ directory.');
    return;
  }

  final vocabContent = vocabFile.readAsStringSync();

  //Initialize the tokenizer
  final tokenizer = BertTokenizer.fromStringContent(vocabContent);

  //Prepare the input for a BERT model
  final text = "Hello worldwide Dart is awesome!";
  final maxLength = 12;

  final input = tokenizer.prepareNerInput(text, maxLength);

  //Print the results
  print('--- BERT Tokenizer Example ---');
  print('Original Text:     $text');

  //Show the intermediate string tokens
  print('Tokenized Strings: ${tokenizer.tokenize(text)}');
  print('------------------------------');

  //Show the final arrays needed for the ML Model
  print('Input IDs:         ${input.inputIds}');
  print('Input Mask:        ${input.inputMask}');
  print('Segment IDs:       ${input.segmentIds}');
}
