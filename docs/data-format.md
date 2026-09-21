# Training data format

any2jev trains on **labelled System One requests**: one JSON object per line, each exactly what a
`/v1/systemone` request looks like, plus a `label` on every question. What the model trains on is
byte-identical to what the server feeds it, so there is no prompt drift between training and serving.

```json
{"state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
 "questions": {
   "department": {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                               "shipping": "Delivery status, delays, lost packages",
                               "billing": "Charges, invoices, payment problems"},
                  "label": "returns"},
   "escalate":   {"type": "noul", "instructions": "Does this need urgent human attention?", "label": false},
   "frustration":{"type": "score", "instructions": "How frustrated is the customer?",
                  "criteria": ["Calm", "Frustrated", "Very angry"], "label": 1}}}
```

| field | rules |
|---|---|
| `state` | string, object or array. Objects and arrays are rendered as `key: value` lines / bullets. |
| `questions` | map of your own ids to questions; 1 to as many as fit the context. |
| `type: noul` | `instructions` (string or object), optional `criteria: {"true": …, "false": …}`; `label`: `true`/`false`. |
| `type: choice` | `criteria`: map of option key to description (or `null`); 1–255 options; `label`: an option key. |
| `type: score` | `criteria`: ordered list of level descriptions (2–255; Jev accepts up to 10); `label`: level index from 0. |

Tips that matter for the model:

* Vary the phrasing of instructions and criteria, wrap the same state as a string sometimes and as a
  JSON object other times. The synthetic generator (`any2jev data synthetic`) does this.
* Include "none of the above" options both as the correct answer and as a distractor, with varied
  wording, or the model learns "that wording ⇒ pick it".
* Keep arithmetic, dates and counting out of the questions; decide them in code (Jev's own docs say the
  same).

## Built-in sources

`any2jev data synthetic` needs no download: support tickets with a department Choice, an urgency
Score, a refund Noul and a tone Choice, with rule-based labels.

`any2jev data build --sources boolq,ag_news,yelp,sst5,mnli,banking77` converts public datasets
(needs `pip install any2jev[data]`):

| source | primitive | options | what it teaches |
|---|---|---|---|
| boolq | Noul | 2 | reading comprehension yes/no |
| ag_news | Choice | 4 (shuffled) | topic classification with descriptions |
| banking77 | Choice | 4–12 sampled of 77 | fine-grained intent among close options |
| mnli | Choice | 3 | entailment / contradiction |
| yelp | Score | 5 | ordered rating |
| sst5 | Score | 5 | ordered sentiment |
