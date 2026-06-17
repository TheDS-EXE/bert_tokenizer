/// Simple BERT tokenizer for NER tasks.
class BertTokenizer {
  late List<String> _vocab;
  late Map<String, int> _vocabMap;
  late List<String> _idsToTokens;

  static const String padToken = '[PAD]';
  static const String unkToken = '[UNK]';
  static const String clsToken = '[CLS]';
  static const String sepToken = '[SEP]';
  static const String maskToken = '[MASK]';

  BertTokenizer._internal();

  /// Creates a tokenizer from a pre‑loaded vocabulary string.
  ///
  /// The [content] should be the raw text of your `vocab.txt` file,
  /// with each token on a new line.
  static BertTokenizer fromStringContent(String content) {
    final tokenizer = BertTokenizer._internal();
    final lines = content.split('\n');
    tokenizer._vocab = [];
    tokenizer._vocabMap = {};

    for (var line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;
      tokenizer._vocab.add(trimmed);
    }

    for (int i = 0; i < tokenizer._vocab.length; i++) {
      tokenizer._vocabMap[tokenizer._vocab[i]] = i;
    }

    if (!tokenizer._vocabMap.containsKey(unkToken) ||
        !tokenizer._vocabMap.containsKey(padToken)) {
      throw FormatException('Vocabulary must contain "$unkToken" and "$padToken".');
    }

    tokenizer._idsToTokens = List.from(tokenizer._vocab);
    return tokenizer;
  }

  /// Tokenizes raw text into subword tokens.
  List<String> tokenize(String text) {
    final words = _splitIntoWords(text.toLowerCase());
    final tokens = <String>[];
    for (var word in words) {
      final subTokens = _wordPieceTokenize(word);
      tokens.addAll(subTokens);
    }
    return tokens;
  }

  List<String> _splitIntoWords(String text) {
    // 1. Add spaces around punctuation marks so they detach from words
    final spacedText = text.replaceAllMapped(
        RegExp(r'([,.!?;:()])'), (match) => ' ${match.group(1)} ');

    // 2. Split everything by whitespace and filter out empty strings
    return spacedText.split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();
  }

  List<String> _wordPieceTokenize(String word) {
    if (_vocabMap.containsKey(word)) return [word];
    final tokens = <String>[];
    int start = 0;
    while (start < word.length) {
      int end = word.length;
      String? curSubstr;
      while (start < end) {
        String substr = (start == 0 ? '' : '##') + word.substring(start, end);
        if (_vocabMap.containsKey(substr)) {
          curSubstr = substr;
          break;
        }
        end--;
      }
      if (curSubstr == null) {
        tokens.add(unkToken);
        start = word.length;
      } else {
        tokens.add(curSubstr);
        start = end;
      }
    }
    return tokens;
  }

  /// Converts a tokenized sequence into input IDs, mask, and segment IDs.
  BertInput prepareNerInput(String text, int maxLength) {
    if (maxLength < 2) {
      throw ArgumentError('maxLength must be at least 2 to accommodate special tokens.');
    }

    final tokens = tokenize(text);

    List<String> truncatedTokens = tokens;
    if (truncatedTokens.length > maxLength - 2) {
      truncatedTokens = truncatedTokens.sublist(0, maxLength - 2);
    }

    List<String> finalTokens = [clsToken, ...truncatedTokens, sepToken];

    final inputIds =
    finalTokens.map((t) => _vocabMap[t] ?? _vocabMap[unkToken]!).toList();

    final attentionMask = List.filled(inputIds.length, 1, growable: true);
    final segmentIds = List.filled(inputIds.length, 0, growable: true);

    final padId = _vocabMap[padToken]!;
    while (inputIds.length < maxLength) {
      inputIds.add(padId);
      attentionMask.add(0);
      segmentIds.add(0);
    }

    return BertInput(
      inputIds: inputIds,
      inputMask: attentionMask,
      segmentIds: segmentIds,
    );
  }

  /// Converts token IDs back to strings.
  List<String> convertIdsToTokens(List<int> ids) {
    return ids.map((id) {
      if (id < 0 || id >= _idsToTokens.length) return unkToken;
      return _idsToTokens[id];
    }).toList();
  }

  int get vocabSize => _vocab.length;
}

/// Input structure for BERT NER model.
class BertInput {
  final List<int> inputIds;
  final List<int> inputMask;
  final List<int> segmentIds;

  BertInput({
    required this.inputIds,
    required this.inputMask,
    required this.segmentIds,
  });
}

/// TOP‑LEVEL FUNCTION for backwards compatibility.
BertInput prepareNerInput(String text, BertTokenizer tokenizer, int maxLength) {
  return tokenizer.prepareNerInput(text, maxLength);
}