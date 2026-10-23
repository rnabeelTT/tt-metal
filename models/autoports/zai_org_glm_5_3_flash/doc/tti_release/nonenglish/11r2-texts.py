"""Stage 11 review #2 (P2 non-English): the teacher-forcing texts for the device A/B and the HF reference.
Usage: 11r2-texts.py OUT_JSON
Items: doc106_s9 = the saved seed-9 doc-106 draw (prompt 65 tokens + the first 38 TT tokens, the same rows as
doc/tti_release/loopdiag/hfloop.json); vi / fa / zh / en = a short user request + a hand-written answer paragraph
(chat template, reasoning_effort low, '</think>' then the paragraph forced)."""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

W = Path.home() / "glm53-flash-bringup"
tok = AutoTokenizer.from_pretrained(str(W / "hfloop_overlay"))


def prompt(msgs):
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, reasoning_effort="low")
    if hasattr(p, "input_ids") or isinstance(p, dict):
        p = p["input_ids"]
    return [int(t) for t in p]


items = []
probe = json.load(open(W / "artifacts/s11/loopprobe/loopprobe.json"))
rec = [r for r in probe["results"] if r["doc"] == "106" and r["cond"] == "base" and r["seed"] == 9][0]
MSGS = json.load(open(W / "artifacts/s11/loopdiag/messages.json"))
p = prompt([{"role": m["role"], "content": m["content"]} for m in MSGS["106"]])
assert len(p) == rec["usage"]["prompt_tokens"] == 65, len(p)
items.append(
    {
        "name": "doc106_s9",
        "lang": "vi",
        "prompt_ids": p,
        "gen_ids": rec["token_ids"][:38],
        "served_p": rec["p"][:38],
        "served_top5": rec["top5"][:38],
    }
)

TEXTS = {
    "vi": (
        "Hãy viết một đoạn văn ngắn giới thiệu về Hà Nội.",
        "Hà Nội là thủ đô của Việt Nam, nằm ở trung tâm đồng bằng sông Hồng. Thành phố có lịch sử hơn một nghìn "
        "năm và nổi tiếng với khu phố cổ, hồ Hoàn Kiếm và nhiều món ăn truyền thống như phở, bún chả và bánh cuốn. "
        "Mỗi buổi sáng, người dân thường tập thể dục quanh hồ, sau đó uống một ly cà phê sữa đá trước khi đi làm. "
        "Vào mùa thu, thời tiết mát mẻ và không khí trong lành, đây là thời điểm đẹp nhất để du khách đến thăm "
        "thành phố.",
    ),
    "fa": (
        "یک پاراگراف کوتاه درباره‌ی تهران بنویس.",
        "تهران پایتخت ایران و بزرگ‌ترین شهر این کشور است. این شهر در دامنه‌ی جنوبی رشته‌کوه البرز قرار دارد و "
        "جمعیت آن بیش از هشت میلیون نفر است. تهران مراکز فرهنگی و تاریخی فراوانی دارد، از جمله کاخ گلستان، "
        "موزه‌ی ملی ایران و بازار بزرگ. در فصل زمستان، بسیاری از مردم برای اسکی به پیست‌های نزدیک شهر می‌روند. "
        "ترافیک سنگین و آلودگی هوا از مشکلات اصلی این شهر به شمار می‌آیند.",
    ),
    "zh": (
        "请写一段简短的文字介绍杭州。",
        "杭州是浙江省的省会，位于中国东南沿海。这座城市以西湖闻名，湖边有许多古老的寺庙、园林和茶园。每年春天，"
        "龙井茶的新叶被采摘下来，经过炒制后成为闻名全国的绿茶。杭州也是一座现代化的城市，许多科技公司在这里设立"
        "总部。游客可以白天游览湖光山色，晚上品尝地道的杭帮菜，比如西湖醋鱼和东坡肉。",
    ),
    "en": (
        "Write a short paragraph about Edinburgh.",
        "Edinburgh is the capital of Scotland and sits on the southern shore of the Firth of Forth. The city is "
        "known for its medieval Old Town and its elegant Georgian New Town, which together form a UNESCO World "
        "Heritage Site. Every August, the Edinburgh Festival Fringe brings thousands of performers and visitors to "
        "the streets. On a clear day, a walk up Arthur's Seat, an ancient volcano in Holyrood Park, rewards hikers "
        "with views across the whole city and out to the sea.",
    ),
}
for lang, (q, a) in TEXTS.items():
    p = prompt([{"role": "user", "content": q}])
    g = tok.encode("</think>" + a, add_special_tokens=False)
    items.append({"name": f"para_{lang}", "lang": lang, "prompt_ids": p, "gen_ids": [int(t) for t in g]})
for it in items:
    print(it["name"], len(it["prompt_ids"]), len(it["gen_ids"]), repr(tok.decode(it["gen_ids"][:6])))
Path(sys.argv[1]).write_text(json.dumps(items, ensure_ascii=False) + "\n")
