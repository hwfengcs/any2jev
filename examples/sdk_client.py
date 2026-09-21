"""Call a local any2jev server with the official TypeSafe SDK: only base_url changes.

    any2jev serve runs/my-model --port 8009
    python examples/sdk_client.py
"""

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="any2jev-latest")

resp = client.system_one(
    state="Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"returns": "Exchanges, refunds, wrong or damaged items",
                      "shipping": "Delivery status, delays, lost packages",
                      "billing": "Charges, invoices, payment problems"},
        ),
        "escalate": Noul(instructions="Does this need urgent human attention?"),
        "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Frustrated", "Very angry"]),
    },
)
d = resp.choices["department"]
print(f"department: {d.choice}  (confidence {d.confidence:.2f})  {d.probabilities}")
print(f"escalate:   p(yes) = {resp.nouls['escalate'].noul:.2f}")
s = resp.scores["frustration"]
print(f"frustration: {s.score:.2f} on {s.legend}")
