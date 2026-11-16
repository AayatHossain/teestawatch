"""Pick radar scenes for each flood event and order HyP3 RTC 10 m processing."""
from pathlib import Path
import pandas as pd
import hyp3_sdk as sdk

ROOT = Path(__file__).resolve().parents[1]
passes = pd.read_csv(ROOT / "data" / "s1_passes.csv")

PLAN = [("dry", f"dry_{d[:4]}", 114, d) for d in
        ["2017-03-07", "2018-03-02", "2019-02-25", "2020-03-03", "2021-02-26",
         "2022-03-05", "2023-02-28", "2024-03-06", "2025-03-01", "2026-02-24"]]
EVENTS = {"flood_2017_08": (114, "2017-07-17", "2017-08-10"),
          "flood_2019_07": (114, "2019-06-25", "2019-07-19"),
          "flood_2020_07": (114, "2020-06-07", "2020-07-13"),
          "flood_2024_07": (114, "2024-06-10", "2024-07-04"),
          "flood_2025_08": (114, "2025-07-23", "2025-08-16"),
          "flood_2025_10": (150, "2025-09-23", "2025-10-05"),
          "flood_2026_07": (114, "2026-06-12", "2026-07-01")}
for ev, (ro, before, after) in EVENTS.items():
    PLAN += [("flood_before", ev, ro, before), ("flood_after", ev, ro, after)]

rows = []
for purpose, event, ro, date in PLAN:
    hit = passes[(passes.rel_orbit == ro) & (passes.date == date)]
    assert len(hit) == 1, (event, date)
    for s in hit.scenes.iloc[0].split(";"):
        rows.append(dict(purpose=purpose, event=event, date=date, rel_orbit=ro, scene=s,
                         pass_cover=hit.pass_cover.iloc[0]))
scenes = pd.DataFrame(rows)
scenes.to_csv(ROOT / "data" / "scenes.csv", index=False)
print(scenes.groupby("purpose").size().to_string(), "\nunique scenes:", scenes.scene.nunique())

hyp3 = sdk.HyP3()
already = {j.job_parameters["granules"][0] for j in hyp3.find_jobs(job_type="RTC_GAMMA")}
todo = [s for s in scenes.scene.unique() if s not in already]
print("already ordered:", len(already), " to order:", len(todo), " credits before:", hyp3.check_credits())
batch = sdk.Batch()
for s in todo:
    batch += hyp3.submit_rtc_job(s, name="teesta", resolution=10)
print("submitted:", len(batch), " credits after:", hyp3.check_credits())
