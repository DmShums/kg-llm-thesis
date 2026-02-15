from modules.mediating_selector import MediatingOntologySelector

o1_path = "data/anatomy/human-mouse/human.owl"
o2_path = "data/anatomy/human-mouse/mouse.owl"

selector = MediatingOntologySelector(
    o1_path,
    o2_path
)

mediators = selector.select_top_mediators(top_k=3, download=True)

print(mediators)