"""
LogMap + BioPortal mediating ontology runner.

Implements:
1: M1 := LogMap(O1, MO)
2: M2 := LogMap(MO, O2)
3: MC := ComposeMappings(M1, M2)

Additionally computes:
MC_minus_direct := MC − LogMap(O1, O2)
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set

from utils.constants import LOGGER
from modules.logmap_wrapper import run_logmap_alignment


class LogMapBioRunner:
    def __init__(
        self,
        o1_path: str,
        o2_path: str,
        mediating_ontology_path: str,
        work_dir: str = "output/logmapbio"
    ):
        self.o1 = o1_path
        self.o2 = o2_path
        self.mo = mediating_ontology_path

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("LogMapBioRunner initialized")

    # ------------------------------------------------------------
    # STEP 1 & 2: Run LogMap pairwise
    # ------------------------------------------------------------

    def _run_pairwise(self, onto_a: str, onto_b: str, subdir: str) -> str:
        out_dir = self.work_dir / subdir
        out_dir.mkdir(exist_ok=True)

        result = run_logmap_alignment(onto_a, onto_b, str(out_dir))

        if not result["success"]:
            raise RuntimeError(f"LogMap failed: {result['error']}")

        LOGGER.info(f"Pairwise alignment completed: {subdir}")
        return result["mappings_rdf"]

    # ------------------------------------------------------------
    # RDF parsing
    # ------------------------------------------------------------

    @staticmethod
    def _parse_rdf_mappings(rdf_path: str) -> Set[Tuple[str, str]]:
        mappings = set()
        e1, e2 = None, None

        with open(rdf_path, "r", encoding="utf-8") as f:
            for line in f:
                if "entity1 rdf:resource=" in line:
                    e1 = line.split('"')[1]
                elif "entity2 rdf:resource=" in line:
                    e2 = line.split('"')[1]

                if e1 and e2:
                    mappings.add((e1, e2))
                    e1, e2 = None, None

        return mappings

    # ------------------------------------------------------------
    # STEP 3: Compose mappings
    # ------------------------------------------------------------

    def _compose_mappings(
        self,
        m1_path: str,
        m2_path: str
    ) -> Set[Tuple[str, str]]:

        LOGGER.info("Composing mappings via mediating ontology")

        m1 = self._parse_rdf_mappings(m1_path)
        m2 = self._parse_rdf_mappings(m2_path)

        # MO -> O2 lookup
        mo_to_o2: Dict[str, Set[str]] = {}
        for mo, o2 in m2:
            mo_to_o2.setdefault(mo, set()).add(o2)

        composed = set()

        for o1, mo in m1:
            if mo in mo_to_o2:
                for o2 in mo_to_o2[mo]:
                    composed.add((o1, o2))

        LOGGER.info(f"Composed {len(composed)} mappings")
        return composed

    # ------------------------------------------------------------
    # Write mappings helper
    # ------------------------------------------------------------

    def _write_txt(
        self,
        mappings: Set[Tuple[str, str]],
        filename: str
    ) -> str:

        out_path = self.work_dir / filename

        with open(out_path, "w", encoding="utf-8") as f:
            for e1, e2 in sorted(mappings):
                f.write(f"{e1}|{e2}|=|1.0\n")

        return str(out_path)

    # ------------------------------------------------------------
    # Direct LogMap alignment
    # ------------------------------------------------------------

    def _run_direct_alignment(self) -> Set[Tuple[str, str]]:

        LOGGER.info("Running direct LogMap(O1, O2)")

        direct_dir = self.work_dir / "direct_o1_o2"
        direct_dir.mkdir(exist_ok=True)

        result = run_logmap_alignment(self.o1, self.o2, str(direct_dir))

        if not result["success"]:
            raise RuntimeError(f"Direct LogMap failed: {result['error']}")

        return self._parse_rdf_mappings(result["mappings_rdf"])

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------
    def run_composition_only(self) -> dict:
        LOGGER.info("Running MC composition only")

        # Step 1
        m1_path = self._run_pairwise(self.o1, self.mo, "step1_o1_mo")

        # Step 2
        m2_path = self._run_pairwise(self.mo, self.o2, "step2_mo_o2")

        # Step 3
        mc = self._compose_mappings(m1_path, m2_path)
        mc_file = self._write_txt(mc, "MC_composed.txt")

        LOGGER.info(f"MC size: {len(mc)}")

        return {
            "success": True,
            "M1_path": m1_path,
            "M2_path": m2_path,
            "MC_file": mc_file,
            "MC_size": len(mc)
        }

    def run(self) -> dict:
        """
        Executes:
        - M1
        - M2
        - MC
        - MC_minus_direct
        """

        LOGGER.info("Starting mediating alignment (composition-only mode)")

        # Step 1
        m1_path = self._run_pairwise(self.o1, self.mo, "step1_o1_mo")

        # Step 2
        m2_path = self._run_pairwise(self.mo, self.o2, "step2_mo_o2")

        # Step 3
        mc = self._compose_mappings(m1_path, m2_path)
        mc_file = self._write_txt(mc, "MC_composed.txt")

        # Direct alignment
        direct_mappings = self._run_direct_alignment()
        direct_file = self._write_txt(direct_mappings, "direct_mappings.txt")

        # Set difference
        mc_minus_direct = mc - direct_mappings
        mc_minus_file = self._write_txt(
            mc_minus_direct,
            "MC_minus_direct.txt"
        )

        LOGGER.info(f"MC size: {len(mc)}")
        LOGGER.info(f"Direct size: {len(direct_mappings)}")
        LOGGER.info(f"MC - Direct size: {len(mc_minus_direct)}")

        return {
            "success": True,
            "M1_path": m1_path,
            "M2_path": m2_path,
            "MC_file": mc_file,
            "Direct_file": direct_file,
            "MC_minus_direct_file": mc_minus_file,
            "MC_size": len(mc),
            "Direct_size": len(direct_mappings),
            "MC_minus_size": len(mc_minus_direct),
        }