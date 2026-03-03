from pathlib import Path

def format_subsets_ontologies_paths(dataset_name: str, set_name: str) -> tuple[Path, Path]:
    onto_data_dir = Path("data") / dataset_name / set_name
    source_ontology, target_ontology = set_name.split("-")

    if dataset_name == "anatomy":
        target_filename = f"{source_ontology}.owl"
        source_filename = f"{target_ontology}.owl"

    if dataset_name == "bioml-2024":
        if "." in target_ontology:
            suffix = target_ontology.split(".")[-1]
            source_ontology = f"{source_ontology}.{suffix}"

        source_filename = f"{source_ontology}.owl"
        target_filename = f"{target_ontology}.owl"

    if dataset_name == "largebio":
        onto_data_dir = Path("data") / dataset_name
        source_filename = f"oaei_{source_ontology.upper()}_whole_ontology.owl"
        target_filename = f"oaei_{target_ontology.upper()}_whole_ontology.owl"

    if dataset_name == "largebio_small":
        source_filename = f"oaei_{source_ontology.upper()}_small_overlapping_{target_ontology}.owl"
        target_filename = f"oaei_{target_ontology.upper()}_small_overlapping_{source_ontology}.owl"

    return onto_data_dir / source_filename, onto_data_dir / target_filename
