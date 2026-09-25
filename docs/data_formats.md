# Data formats

## Inventory

The demo [sense_inventory.tsv](../examples/sense_inventory.tsv) contains `sense_id, lemma, upos, definition`. CSV, TSV and Excel are supported. Excel input reads the first sheet; export the intended inventory sheet explicitly if necessary. Duplicate normalized sense IDs overwrite earlier rows, so prepare one definition per sense ID.

## WebAnno annotations

The reader expects columns in this order: token ID; character offsets; token text; POS; UPOS; named-entity ID; named-entity type; lemma; MWE ID; MWE lemma; MWE type; comment; explanation; KBid; number of senses; origin; Possible. Older exports without the comment field are also supported. `#Text=` introduces the sentence. Candidate IDs in Possible are separated by semicolons.

Offsets are zero-based, end-exclusive offsets into the sentence string. A multiword target combines the spans sharing its MWE ID. Identifiers are normalized, including WordNet URLs and dictionary sense suffixes. Do not strip slash suffixes from dictionary identifiers.

## Training pairs

Each JSONL object contains `text, definition, label, sentence, target, sense_id, source_file`. Target spans are marked with double asterisks. The selected sense receives label 1.0; other candidate definitions receive 0.0. Candidates lacking definitions are exported as coverage errors.

Gold exclusion uses exact sentence-text equality across the four recognized gold filenames. Preserve sentence text normalization between silver and gold. Check the printed counts before training: wrong or missing gold filenames would prevent the expected exclusions.

## Saved evaluation output

TSV predictions contain `sentence, target, gold_sense_id, ranked_sense_ids, covered`. The evaluator writes ranked IDs separated by semicolons. The toy example additionally supplies the equivalent objects as JSONL, to make offline scoring easy to inspect.
