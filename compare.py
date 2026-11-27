import json

with (
    open("baseline.json", encoding="utf-8") as f1,
    open("candidatev2.json", encoding="utf-8") as f2,
):
  base = {c["input"]: c for c in json.load(f1)["test_cases"]}
  cand = {c["input"]: c for c in json.load(f2)["test_cases"]}

print(
    f"{'Q#':<4} {'Type':<14} {'Base Rec':<10} {'CandV2 Rec':<12}"
    f" {'CandV2 Prec':<12} Question"
)
for q, c in cand.items():
  b_rec = base[q]["metrics"]["Contextual Recall"]["score"]
  c_rec = c["metrics"]["Contextual Recall"]["score"]
  c_prec = c["metrics"]["Contextual Precision"]["score"]
  if c_rec < b_rec or c_rec < 1.0:
    print(
        f"{c['index']:<4} {str(c.get('question_type', '')):<14} {b_rec:<10.2f}"
        f" {c_rec:<12.2f} {c_prec:<12.2f} {q[:55]}"
    )