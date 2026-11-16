"""Build the browser labelling pages, one per labeller."""
import json
import pandas as pd
from common import ROOT

OUT = ROOT / "evaluation" / "hand_labels"
tpl = (ROOT / "src" / "label_tool_template.html").read_text(encoding="utf-8")
assert tpl.count("/*__TILES__*/null") == 1 and tpl.count('/*__PART__*/""') == 1
payload = {r["id"]: r for r in json.load(open(OUT / "tiles_payload.json"))}
tiles = pd.read_csv(OUT / "tiles.csv")
tiles["kind0"] = tiles.kind.str.replace(" (top-up)", "", regex=False)
order = tiles.sort_values(["date", "kind0", "tile_id"]).tile_id.tolist()
person = {tid: 1 + (i % 2) for i, tid in enumerate(order)}
tiles["person"] = tiles.tile_id.map(person)
tiles.drop(columns="kind0").to_csv(OUT / "tiles.csv", index=False)
old = OUT / "teesta_labeller.html"
if old.exists():
    old.unlink()
for p in (1, 2):
    ids = tiles[tiles.person == p].tile_id.tolist()
    html = tpl.replace("/*__TILES__*/null", json.dumps([payload[i] for i in ids], separators=(",", ":")))               .replace('/*__PART__*/""', json.dumps(f"Person {p}"))
    f = OUT / f"teesta_labeller_person{p}.html"
    f.write_text(html, encoding="utf-8")
    sub = tiles[tiles.person == p]
    print(f"{f.name}: {len(ids)} tiles, {len(html)/1e6:.1f} MB | per date {sub.groupby('date').size().to_dict()} | kinds {sub.kind.value_counts().to_dict()} | with optical {sum('optical' in payload[i] for i in ids)}")
