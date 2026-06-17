import 'package:test/test.dart';
import 'package:bert_tokenizer/bert_tokenizer.dart';

void main() {
  // A tiny, predictable vocabulary for testing
  const mockVocab = '''
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
hello
world
##wide
dart
is
awesome
un
##believable
.
!
''';

  late BertTokenizer tokenizer;

  // This runs once before all the tests
  setUpAll(() {
    tokenizer = BertTokenizer.fromStringContent(mockVocab);
  });

  group('Initialization and Vocab', () {
    test('Vocab size should be correct', () {
      // 5 special tokens + 10 words/punctuation = 15
      expect(tokenizer.vocabSize, 15);
    });

    test('Throws FormatException if required tokens are missing', () {
      const badVocab = '''
[CLS]
[SEP]
hello
world
''';
      expect(
            () => BertTokenizer.fromStringContent(badVocab),
        throwsFormatException,
      );
    });
  });

  group('String Tokenization', () {
    test('Splits standard words correctly', () {
      final tokens = tokenizer.tokenize('hello world');
      expect(tokens, ['hello', 'world']);
    });

    test('Handles subwords (WordPiece) correctly', () {
      final tokens = tokenizer.tokenize('worldwide unbelievable');
      expect(tokens, ['world', '##wide', 'un', '##believable']);
    });

    test('Handles punctuation', () {
      final tokens = tokenizer.tokenize('hello world!');
      expect(tokens, ['hello', 'world', '!']);
    });

    test('Replaces unknown words with [UNK]', () {
      final tokens = tokenizer.tokenize('hello alien');
      expect(tokens, ['hello', '[UNK]']);
    });

    test('Converts text to lowercase', () {
      final tokens = tokenizer.tokenize('HELLO');
      expect(tokens, ['hello']);
    });
  });

  group('BERT Input Preparation (prepareNerInput)', () {
    test('Generates correct arrays with padding', () {
      // "hello world" -> [CLS] hello world [SEP] [PAD] [PAD] ...
      final input = tokenizer.prepareNerInput('hello world', 6);

      // IDs: [CLS]=2, hello=5, world=6, [SEP]=3, [PAD]=0
      expect(input.inputIds, [2, 5, 6, 3, 0, 0]);

      // Mask: 1 for real tokens, 0 for padding
      expect(input.inputMask, [1, 1, 1, 1, 0, 0]);

      // Segment IDs: all 0s for single sequence
      expect(input.segmentIds, [0, 0, 0, 0, 0, 0]);
    });

    test('Truncates exactly to maxLength if input is too long', () {
      // Text has 5 words. With CLS and SEP, it needs 7 slots.
      // We force a max length of 5.
      final input =
      tokenizer.prepareNerInput('hello worldwide dart is awesome', 5);

      expect(input.inputIds.length, 5);
      expect(input.inputMask.length, 5);
      expect(input.segmentIds.length, 5);

      // The last token MUST be [SEP] (ID 3), even if truncated
      expect(input.inputIds.last, 3);
      // The first token MUST be [CLS] (ID 2)
      expect(input.inputIds.first, 2);
    });

    test('Throws ArgumentError if maxLength is less than 2', () {
      expect(
            () => tokenizer.prepareNerInput('hello world', 1),
        throwsArgumentError,
      );

      expect(
            () => tokenizer.prepareNerInput('hello world', 0),
        throwsArgumentError,
      );
    });

    test('Truncates to just special tokens when maxLength is 2', () {
      final input = tokenizer.prepareNerInput('hello worldwide dart', 2);

      expect(input.inputIds.length, 2);
      expect(input.inputIds, [2, 3]); // [CLS] and [SEP]
    });
  });

  group('ID to Token Conversion', () {
    test('Converts IDs back to string tokens', () {
      final ids = [2, 5, 6, 3];
      final strings = tokenizer.convertIdsToTokens(ids);

      expect(strings, ['[CLS]', 'hello', 'world', '[SEP]']);
    });

    test('Returns [UNK] for out-of-bounds IDs', () {
      final ids = [2, 999, 3];
      final strings = tokenizer.convertIdsToTokens(ids);

      expect(strings, ['[CLS]', '[UNK]', '[SEP]']);
    });
  });
}