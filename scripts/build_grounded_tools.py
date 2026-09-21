"""Two families the mix never had: is this claim supported, and does a tool fit.

`build_unseen_eval.py` found the gap. On HaluEval the 1.7B and the 8B score the same
(.73, .72 against Jev's .91), and on BFCL tool relevance the 1.7B is near a coin. Same
score at two sizes means it is the data. Nothing in the mix asks either question.

    uv run python scripts/build_grounded_tools.py
    # data/grounded_tools_train.jsonl (3,000) and data/grounded_tools_eval.jsonl (600)

Every label is a human's. Most public function-calling sets are written by a language
model, requests included, which the no-teacher rule excludes; so requests here are
utterances people wrote for intent datasets, and the tool specs they are matched
against are written out below. HaluEval and BFCL supply nothing: they are the test.

  claim_support    VitaminC     evidence + claim -> supported?              noul    CC BY-SA 3.0
  claim_status     VitaminC     supports / refutes / not enough information choice
  answer_support   SQuAD v2     passage + question + answer -> supported?   noul    CC BY-SA 4.0
  tool_relevance   CLINC, SNIPS request + tools -> can any of them do it?   noul    CC BY 3.0, CC0
  tool_choice      CLINC, SNIPS request + tools -> which one, or none       choice
  tool_call_check  CLINC, SNIPS request + proposed tool -> the right one?   noul

Negatives are the hard kind. A request whose tool was taken out of the list still sees
its neighbours (pay_bill without pay_bill, but with bill_balance and transfer); CLINC's
own out-of-scope utterances are requests no listed tool handles; an unanswerable SQuAD
question is paired with a span from the same passage, so the answer is in the text and
still not supported.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hf_eval import item, onehot

# intent -> (group, tool name, description, parameters). Groups supply the near misses.
TOOLS: dict[str, tuple[str, str, str, list[str]]] = {
    # banking
    "transfer": ("bank", "bank.transfer_funds", "Move money between two of the user's accounts.", ["from_account", "to_account", "amount"]),
    "pay_bill": ("bank", "bank.pay_bill", "Pay a bill to a named payee from a chosen account.", ["payee", "amount", "account"]),
    "balance": ("bank", "bank.get_balance", "Return the current balance of an account.", ["account"]),
    "bill_balance": ("bank", "bank.get_bill_amount", "Return how much is owed on a bill.", ["payee"]),
    "bill_due": ("bank", "bank.get_bill_due_date", "Return the date a bill is due.", ["payee"]),
    "transactions": (
        "bank",
        "bank.list_transactions",
        "List recent transactions on an account, optionally filtered by merchant or date.",
        ["account", "since", "merchant"],
    ),
    "spending_history": ("bank", "bank.spending_summary", "Summarise how much was spent in a category over a period.", ["category", "period"]),
    "freeze_account": ("bank", "bank.freeze_account", "Freeze an account so nothing can be charged to it.", ["account"]),
    "report_lost_card": ("bank", "cards.report_lost", "Report a card as lost or stolen and block it.", ["card"]),
    "report_fraud": ("bank", "cards.report_fraud", "Dispute a transaction the user does not recognise.", ["transaction_id"]),
    "new_card": ("bank", "cards.request_new", "Apply for a new credit or debit card.", ["card_type"]),
    "damaged_card": ("bank", "cards.replace_damaged", "Order a replacement for a card that no longer works.", ["card"]),
    "pin_change": ("bank", "cards.change_pin", "Set a new PIN for a card.", ["card", "new_pin"]),
    "credit_limit": ("bank", "cards.get_credit_limit", "Return the credit limit on a card.", ["card"]),
    "credit_limit_change": ("bank", "cards.request_limit_change", "Ask for a card's credit limit to be raised or lowered.", ["card", "new_limit"]),
    "credit_score": ("bank", "credit.get_score", "Return the user's current credit score.", []),
    "order_checks": ("bank", "bank.order_checkbook", "Order a new book of checks for an account.", ["account"]),
    "exchange_rate": ("bank", "fx.get_rate", "Return the exchange rate between two currencies.", ["from_currency", "to_currency"]),
    "interest_rate": ("bank", "bank.get_interest_rate", "Return the interest rate on an account.", ["account"]),
    "direct_deposit": ("bank", "payroll.setup_direct_deposit", "Set up or explain direct deposit of pay into an account.", ["account"]),
    # travel
    "book_flight": ("travel", "flights.book", "Book a flight between two cities on given dates.", ["origin", "destination", "depart_date", "return_date"]),
    "flight_status": ("travel", "flights.get_status", "Return whether a flight is on time, delayed or cancelled.", ["flight_number"]),
    "book_hotel": ("travel", "hotels.book", "Reserve a hotel room in a city for given dates.", ["city", "check_in", "check_out"]),
    "car_rental": ("travel", "cars.rent", "Rent a car at a location for given dates.", ["location", "pickup_date", "dropoff_date"]),
    "lost_luggage": ("travel", "airline.report_lost_luggage", "File a report for baggage that did not arrive.", ["flight_number"]),
    "travel_alert": ("travel", "travel.get_advisories", "Return safety advisories for a country.", ["country"]),
    "international_visa": ("travel", "travel.visa_requirements", "Say whether a visa is needed to visit a country.", ["country"]),
    "uber": ("travel", "rides.request", "Request a ride to a destination.", ["destination", "passengers"]),
    "directions": ("travel", "maps.get_directions", "Return directions from one place to another.", ["origin", "destination"]),
    "distance": ("travel", "maps.get_distance", "Return how far it is, or how long it takes, between two places.", ["origin", "destination"]),
    "traffic": ("travel", "maps.get_traffic", "Return current traffic on the way to a place.", ["destination"]),
    "share_location": ("travel", "location.share", "Send the user's current location to a contact.", ["contact"]),
    # food
    "restaurant_reservation": ("food", "restaurants.reserve", "Book a table at a restaurant for a party at a time.", ["restaurant", "party_size", "time"]),
    "cancel_reservation": ("food", "restaurants.cancel_reservation", "Cancel an existing restaurant booking.", ["restaurant"]),
    "restaurant_suggestion": ("food", "restaurants.search", "Suggest restaurants by cuisine or area.", ["cuisine", "near"]),
    "restaurant_reviews": ("food", "restaurants.get_reviews", "Return reviews for a named restaurant.", ["restaurant"]),
    "recipe": ("food", "recipes.find", "Find a recipe for a dish.", ["dish"]),
    "calories": ("food", "nutrition.get_calories", "Return the calories in a food.", ["food"]),
    "ingredients_list": ("food", "recipes.get_ingredients", "List the ingredients a dish needs.", ["dish"]),
    "BookRestaurant": (
        "food",
        "dining.book_table",
        "Reserve a table, given a place or cuisine, a party size and a time.",
        ["restaurant_or_cuisine", "party_size", "time", "location"],
    ),
    # productivity
    "alarm": ("time", "clock.set_alarm", "Set an alarm for a time of day.", ["time", "label"]),
    "timer": ("time", "clock.start_timer", "Start a countdown timer for a duration.", ["duration"]),
    "reminder_update": ("time", "reminders.create", "Create a reminder to do something at a time.", ["text", "time"]),
    "reminder": ("time", "reminders.list", "Read back the reminders the user has set.", []),
    "calendar_update": ("time", "calendar.add_event", "Add an event to the calendar.", ["title", "start", "end"]),
    "calendar": ("time", "calendar.get_events", "Read what is on the calendar for a day.", ["date"]),
    "schedule_meeting": ("time", "meetings.schedule", "Schedule a meeting with people at a time and book a room.", ["attendees", "time", "room"]),
    "todo_list_update": ("time", "todo.add_item", "Add or remove an item on the to-do list.", ["item", "action"]),
    "shopping_list_update": ("time", "shopping.add_item", "Add or remove an item on the shopping list.", ["item", "action"]),
    "pto_request": ("time", "hr.request_time_off", "Submit a request for paid time off between two dates.", ["start_date", "end_date"]),
    # media and device
    "play_music": ("media", "music.play", "Play a song, artist, album or playlist.", ["query"]),
    "PlayMusic": (
        "media",
        "player.play_track",
        "Play music by artist, track, album, genre or year on a chosen service.",
        ["artist", "track", "service", "year"],
    ),
    "AddToPlaylist": ("media", "playlists.add_track", "Add a song or an artist to one of the user's playlists.", ["item", "playlist"]),
    "update_playlist": ("media", "playlists.update", "Add the current or a named song to a playlist.", ["song", "playlist"]),
    "next_song": ("media", "music.skip", "Skip to the next song.", []),
    "what_song": ("media", "music.identify", "Say which song is playing.", []),
    "change_volume": ("media", "device.set_volume", "Turn the speaker volume up or down.", ["level"]),
    "SearchCreativeWork": ("media", "catalog.search_work", "Find a book, film, show, game or album by title.", ["title", "type"]),
    "SearchScreeningEvent": ("media", "cinema.find_showtimes", "Find when and where a film is showing.", ["film", "location", "time"]),
    "RateBook": ("media", "books.rate", "Give a book a rating out of some number of points or stars.", ["title", "rating", "scale"]),
    "smart_home": ("media", "home.control_device", "Turn a connected device on or off, or change its setting.", ["device", "action"]),
    "make_call": ("media", "phone.call", "Place a phone call to a contact or number.", ["contact"]),
    "text": ("media", "phone.send_text", "Send a text message to a contact.", ["contact", "message"]),
    "find_phone": ("media", "phone.locate", "Make the user's phone ring so they can find it.", []),
    # lookups
    "weather": ("lookup", "weather.get_forecast", "Return the weather for a place and day.", ["location", "date"]),
    "GetWeather": (
        "lookup",
        "forecast.lookup",
        "Return the forecast or a condition such as rain or temperature for a place and time.",
        ["location", "time", "condition"],
    ),
    "translate": ("lookup", "language.translate", "Translate a phrase into another language.", ["text", "target_language"]),
    "definition": ("lookup", "dictionary.define", "Return the meaning of a word.", ["word"]),
    "spelling": ("lookup", "dictionary.spell", "Spell a word.", ["word"]),
    "calculator": ("lookup", "math.evaluate", "Work out an arithmetic expression.", ["expression"]),
    "measurement_conversion": ("lookup", "units.convert", "Convert an amount from one unit to another.", ["amount", "from_unit", "to_unit"]),
    "time": ("lookup", "clock.get_time", "Return the current time in a place.", ["location"]),
    "timezone": ("lookup", "clock.get_timezone", "Return the time zone a place is in.", ["location"]),
    "order_status": ("lookup", "orders.track", "Return where an order is and when it will arrive.", ["order_id"]),
    "order": ("lookup", "orders.place", "Buy an item online.", ["item", "quantity"]),
    # car
    "schedule_maintenance": ("car", "garage.book_service", "Book the car in for maintenance.", ["service", "date"]),
    "tire_pressure": ("car", "car.get_tire_pressure", "Return the car's tyre pressure.", []),
    "gas": ("car", "car.get_fuel_level", "Return how much fuel is in the tank.", []),
    "mpg": ("car", "car.get_fuel_economy", "Return the car's fuel economy.", ["model"]),
    "oil_change_when": ("car", "car.next_oil_change", "Say when the next oil change is due.", []),
}

RELEVANCE = [
    "Can at least one of the listed `tools` carry out the user's `request`?",
    "Is there a tool in `tools` that does what `request` asks for?",
    "Would calling one of these `tools` be a correct way to handle `request`?",
]
RELEVANCE_CRIT = {"true": "A listed tool does what the request asks for", "false": "None of the listed tools fits; calling any of them would be wrong"}
CHOOSE = ["Which tool should handle the user's request? Pick `none` if no listed tool fits.", "Pick the tool that does what the user asked, or `none`."]
CHECK = [
    "Is `proposed_tool` the right tool for the user's `request`?",
    "An agent is about to call `proposed_tool` for this `request`. Is that the correct tool?",
]
CHECK_CRIT = {"true": "The tool does what the request asks", "false": "The tool does something else"}

SUPPORT = [
    "Does `evidence` support `claim`?",
    "Is `claim` true according to `evidence`? Anything the evidence does not establish counts as not supported.",
    "Going only on `evidence`, can `claim` be stated as fact?",
]
SUPPORT_CRIT = {"true": "The evidence establishes the claim", "false": "The evidence contradicts the claim or does not settle it"}
STATUS = "What does `evidence` do to `claim`?"
STATUS_KEYS = {"SUPPORTS": "supports", "REFUTES": "refutes", "NOT ENOUGH INFO": "not_enough_information"}
STATUS_CRIT = {
    "supports": "The evidence establishes the claim",
    "refutes": "The evidence shows the claim is false",
    "not_enough_information": "The evidence neither establishes nor rules out the claim",
}
ANSWER = ["Is `answer` a correct answer to `question` according to `passage`?", "Does `passage` support giving `answer` in reply to `question`?"]
ANSWER_CRIT = {"true": "The passage supports the answer", "false": "The passage does not support that answer to that question"}


def spec(intent: str) -> dict:
    _, name, description, parameters = TOOLS[intent]
    return {"name": name, "description": description, "parameters": parameters}


def neighbours(intent: str, rng: random.Random, k: int) -> list[str]:
    """Other tools, same group first: a near miss is what makes the question worth asking."""
    group = TOOLS[intent][0] if intent in TOOLS else None
    near = [t for t in TOOLS if t != intent and TOOLS[t][0] == group]
    far = [t for t in TOOLS if t != intent and TOOLS[t][0] != group]
    rng.shuffle(near)
    rng.shuffle(far)
    take = near[: max(1, k - 1)] + far
    return take[:k]


def yn(ok: bool) -> dict[str, float]:
    return {"yes": float(ok), "no": float(not ok)}


def tools_family(utterances: list[tuple[str, str]], strangers: list[str], n: int, rng: random.Random, heldout: bool) -> list[dict]:
    """`utterances` are (text, intent) with a tool spec; `strangers` are requests no tool here handles."""
    out: list[dict] = []
    third = n // 3
    rng.shuffle(utterances)
    rng.shuffle(strangers)
    pool, odd = iter(utterances), iter(strangers)

    for i in range(third):  # tool_relevance
        q = {"type": "noul", "instructions": rng.choice(RELEVANCE), "criteria": RELEVANCE_CRIT}
        k = rng.choice([1, 1, 2, 3, 4])
        if i % 2 == 0:
            text, intent = next(pool)
            names = [intent] + neighbours(intent, rng, k - 1)
            ok = True
        elif i % 4 == 1:
            text, names, ok = next(odd), rng.sample(list(TOOLS), k), False
        else:
            text, intent = next(pool)
            names, ok = neighbours(intent, rng, k), False
        rng.shuffle(names)
        out.append(item("tool_relevance", "noul", {"request": text, "tools": [spec(t) for t in names]}, q, yn(ok), heldout))

    for i in range(third):  # tool_choice
        k = rng.choice([3, 4, 5, 6])
        if i % 4 == 3:
            text, intent = next(pool)
            names, gold = neighbours(intent, rng, k), "none"
        elif i % 4 == 2:
            text, names, gold = next(odd), rng.sample(list(TOOLS), k), "none"
        else:
            text, intent = next(pool)
            names, gold = [intent] + neighbours(intent, rng, k - 1), TOOLS[intent][1]
        rng.shuffle(names)
        crit = {TOOLS[t][1]: TOOLS[t][2] for t in names} | {"none": "No listed tool does what was asked"}
        q = {"type": "choice", "instructions": rng.choice(CHOOSE), "criteria": crit}
        out.append(item("tool_choice", "choice", text, q, onehot(list(crit), gold), heldout))

    for i in range(n - 2 * third):  # tool_call_check
        q = {"type": "noul", "instructions": rng.choice(CHECK), "criteria": CHECK_CRIT}
        text, intent = next(pool)
        ok = i % 2 == 0
        proposed = intent if ok else neighbours(intent, rng, 1)[0]
        out.append(item("tool_call_check", "noul", {"request": text, "proposed_tool": spec(proposed)}, q, yn(ok), heldout))
    return out


def grounded_family(vitc, squad, n: int, rng: random.Random, heldout: bool) -> list[dict]:
    out: list[dict] = []
    third = n // 3
    rows = [r for r in vitc if r["label"] in STATUS_KEYS and r["evidence"] and len(r["evidence"]) < 1200]
    rng.shuffle(rows)
    by = {k: [r for r in rows if r["label"] == k] for k in STATUS_KEYS}

    # claim_support: half supported; the rest split between refuted and unsettled
    picks = by["SUPPORTS"][: third // 2] + by["REFUTES"][: third // 4] + by["NOT ENOUGH INFO"][: third - third // 2 - third // 4]
    for r in picks:
        q = {"type": "noul", "instructions": rng.choice(SUPPORT), "criteria": SUPPORT_CRIT}
        out.append(item("claim_support", "noul", {"evidence": r["evidence"], "claim": r["claim"]}, q, yn(r["label"] == "SUPPORTS"), heldout))

    # claim_status: a third each, from rows the noul items did not use
    q = {"type": "choice", "instructions": STATUS, "criteria": STATUS_CRIT}
    for k, skip in (("SUPPORTS", third // 2), ("REFUTES", third // 4), ("NOT ENOUGH INFO", third)):
        for r in by[k][skip : skip + third // 3]:
            out.append(item("claim_status", "choice", {"evidence": r["evidence"], "claim": r["claim"]}, q, onehot(list(STATUS_CRIT), STATUS_KEYS[k]), heldout))

    # answer_support: gold spans, swapped spans, and spans offered for questions the passage cannot answer
    by_ctx: dict[str, list[dict]] = {}
    for r in squad:
        if len(r["context"]) < 1400:
            by_ctx.setdefault(r["context"], []).append(r)
    ctxs = [c for c, rs in by_ctx.items() if sum(1 for r in rs if r["answers"]["text"]) >= 2]
    rng.shuffle(ctxs)
    want = n - len(out)
    for i, c in enumerate(ctxs):
        if want <= 0:
            break
        rs = by_ctx[c]
        answered = [r for r in rs if r["answers"]["text"]]
        unanswerable = [r for r in rs if not r["answers"]["text"]]
        r = rng.choice(answered)
        gold = r["answers"]["text"][0]
        others = [o["answers"]["text"][0] for o in answered if o["answers"]["text"][0].lower() not in {a.lower() for a in r["answers"]["text"]}]
        if i % 2 == 0:
            question, answer, ok = r["question"], gold, True
        elif unanswerable and i % 4 == 1:
            question, answer, ok = rng.choice(unanswerable)["question"], gold, False
        elif others:
            question, answer, ok = r["question"], rng.choice(others), False
        else:
            continue
        q = {"type": "noul", "instructions": rng.choice(ANSWER), "criteria": ANSWER_CRIT}
        out.append(item("answer_support", "noul", {"passage": c, "question": question, "answer": answer}, q, yn(ok), heldout))
        want -= 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="training items per top-level family")
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--train", default="data/grounded_tools_train.jsonl")
    ap.add_argument("--eval", default="data/grounded_tools_eval.jsonl")
    args = ap.parse_args()

    for split, n, path, heldout, seed in (("train", args.n, args.train, False, args.seed), ("eval", args.eval_n, args.eval, True, args.seed + 1)):
        rng = random.Random(seed)
        clinc = load_dataset("clinc/clinc_oos", "plus", split="train" if split == "train" else "test")
        names = clinc.features["intent"].names
        snips = load_dataset("benayas/snips", split="train" if split == "train" else "test")
        utterances = [(r["text"], names[r["intent"]]) for r in clinc if names[r["intent"]] in TOOLS] + [
            (r["text"], r["category"]) for r in snips if r["category"] in TOOLS
        ]
        strangers = [r["text"] for r in clinc if names[r["intent"]] == "oos"]
        vitc = load_dataset("tals/vitaminc", split="train" if split == "train" else "test").shuffle(seed=seed).select(range(20000))
        squad = load_dataset("rajpurkar/squad_v2", split="train" if split == "train" else "validation")
        out = tools_family(utterances, strangers, n, rng, heldout) + grounded_family(vitc, squad, n, rng, heldout)
        rng.shuffle(out)
        with open(path, "w") as f:
            for it in out:
                f.write(json.dumps(it) + "\n")
        print(f"{split}: wrote {len(out)} to {path}")


if __name__ == "__main__":
    main()
