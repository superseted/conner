from conner import ConNER

model = ConNER.load_model("saved_models/conner")

# Test on examples
examples = [
    "Microeconomics focuses on individual markets and consumer behavior.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "In psychology, cognitive dissonance describes the mental stress from holding contradictory beliefs.",
    "A Supply Chain is a network between a company and its suppliers to produce and distribute a specific product.",
    "The human brain is the most complex organ in the body.",
    "Minecraft is a sandbox video game developed by Mojang Studios.",
    "Demand curve - graph of the relationship between the price of a good and the quantity demanded when all other, variables affecting demand constant.",
    "Individual supply is the amount that one particular firm is willing and able to sell.",
    "The Theory of Relativity revolutionized our understanding of space and time.",
    "Law of supply - other thing being equal, the quantity supplied of a goodrises when the price of the goodrises.",
    "It can be found by adding horizontally the individual supply curves. It shows how the total quantity supplied of a good varies as the price of the good varies.",
]


def display_prediction(text):
    concepts = model.extract_concepts(text)
    print(f"{text} -> {concepts}")


for text in examples:
    display_prediction(text)

while True:
    display_prediction(input("Enter a passage: "))
