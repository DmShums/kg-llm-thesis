"""
LogMap matcher and repair as callable tools for the alignment agent.
Runs LogMap via Docker; returns paths to RDF and reduced mapping file.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from utils.constants import LOGGER

from dotenv import load_dotenv
load_dotenv()

WORKSPACE_ROOT = Path(
    os.environ["WORKSPACE_ROOT"]
) if "WORKSPACE_ROOT" in os.environ else Path(__file__).resolve().parent.parent

LOGMAP_JAR = WORKSPACE_ROOT / "modules" / "logmap" / "logmap-matcher-4.0.jar"


def run_logmap_alignment(
    o1_path: str,
    o2_path: str,
    output_dir: str,
    workspace_root: Optional[str] = None,
) -> dict:
    """
    Run LogMap MATCHER on source and target ontologies. Writes RDF and text
    outputs into output_dir.

    Args:
        o1_path: Path to source ontology (relative to workspace or absolute).
        o2_path: Path to target ontology.
        output_dir: Directory for LogMap outputs (e.g. output/).
        workspace_root: Root directory mounted in Docker; default is project root.

    Returns:
        dict with keys: success (bool), mappings_rdf (str path), reduced_txt (str or None),
        error (str or None).
    """
    workspace = workspace_root or str(WORKSPACE_ROOT)
    # Use paths relative to workspace so Docker sees them
    o1_rel = o1_path if not os.path.isabs(o1_path) else os.path.relpath(o1_path, workspace)
    o2_rel = o2_path if not os.path.isabs(o2_path) else os.path.relpath(o2_path, workspace)
    out_rel = output_dir if not os.path.isabs(output_dir) else os.path.relpath(output_dir, workspace)
    os.makedirs(os.path.join(workspace, out_rel), exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workspace}:/workspace",
        "-w", "/workspace",
        "amazoncorretto:8-alpine",
        "java", "-Xmx10g", "-jar", "modules/logmap/logmap-matcher-4.0.jar", # make sure container will allocate at least 10gb of ram
        "MATCHER",
        f"file:{o1_rel}",
        f"file:{o2_rel}",
        out_rel + "/",
        "true",
    ]
    LOGGER.info("Running LogMap MATCHER: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=workspace, timeout=600)
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "mappings_rdf": None,
            "reduced_txt": None,
            "error": f"LogMap exited with code {e.returncode}",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "mappings_rdf": None,
            "reduced_txt": None,
            "error": "Docker or java not found",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "mappings_rdf": None,
            "reduced_txt": None,
            "error": "LogMap timed out (600s)",
        }

    mappings_rdf = os.path.join(workspace, out_rel, "logmap_mappings.rdf")
    reduced_txt = os.path.join(workspace, out_rel, "logmap_mappings_to_ask_oracle_user_llm.txt")
    if not os.path.isfile(mappings_rdf):
        return {
            "success": False,
            "mappings_rdf": None,
            "reduced_txt": reduced_txt if os.path.isfile(reduced_txt) else None,
            "error": "logmap_mappings.rdf not produced",
        }
    return {
        "success": True,
        "mappings_rdf": mappings_rdf,
        "reduced_txt": reduced_txt if os.path.isfile(reduced_txt) else None,
        "error": None,
    }