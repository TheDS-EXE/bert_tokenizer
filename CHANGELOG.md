## 1.1.0
* Fixed Truncation Crash: Rewrote the sequence length handling in prepareNerInput to prevent unhandled StateError exceptions when maxLength is set to small values (like 0 or 1).

* Preserved Special Tokens: Adjusted the truncation logic to trim the raw word pieces before injecting the [CLS] and [SEP] tokens, ensuring the structural markers required by BERT models are never overwritten.

* Enforced Minimum Bounds: Added an ArgumentError validation check to prepareNerInput ensuring maxLength is at least 2, which is the mathematical minimum required to contain the mandatory start and end tokens.

* Added Initialization Validation: Updated fromStringContent to verify that both [UNK] and [PAD] are explicitly defined within the parsed vocabulary map, replacing late-stage runtime crashes (!) with a descriptive FormatException during setup.

## 1.0.2
* Update pub.dev visibility.

## 1.0.1
* Fix documentation links and update repository visibility.

## 1.0.0
* Initial release of pure Dart BERT tokenizer.