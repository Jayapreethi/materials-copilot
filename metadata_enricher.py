#!/usr/bin/env python
"""
Chemistry-Aware Metadata Enricher for CO2M Vector Database

Extracts structured metadata from text including:
- Material identity (name, class, formula, composition)
- Structural properties (surface area, pore size, density)
- Experimental conditions (temperature, pressure, humidity, pH)
- Performance parameters (CO2 uptake, selectivity, stability)
- Method details (experimental vs simulated, measurement method)

Run: python metadata_enricher.py (with enhance() method)
"""

import re
import logging
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataEnricher:
    """Extract and structure chemistry metadata from text."""

    # Material class keywords
    MATERIAL_PATTERNS = {
        "MOF": r"\b(?:metal.?organic.?framework|MOF|UiO|MIL|HKUST|ZIF|IRMOF)\b",
        "Zeolite": r"\b(?:zeolite|13X|5A|mordenite|faujasite)\b",
        "Activated Carbon": r"\b(?:activated.?carbon|AC\b|charcoal|carbon.?foam)\b",
        "Polymer": r"\b(?:polymer|polyamide|polystyrene|PES|PS\b|polymeric)\b",
        "Silica": r"\b(?:silica|SiO2|MCM|SBA|mesoporous.?silica)\b",
        "Graphene": r"\b(?:graphene|GO\b|rGO|graphene.?oxide)\b",
        "Aerogel": r"\b(?:aerogel|cryogel|xerogel)\b",
        "Resin": r"\b(?:resin|ion.?exchange|IER)\b",
        "Composite": r"\b(?:composite|hybrid|nanocomposite|mixed.?matrix)\b",
    }

    # Measurement types
    MEASUREMENT_PATTERNS = {
        "CO2 uptake": r"\bCO2?\s*(?:uptake|adsorption|capture|absorption)\b",
        "N2 uptake": r"\bN2\s*(?:uptake|adsorption)\b",
        "CH4 uptake": r"\bCH4\s*(?:uptake|adsorption)\b",
        "Surface area": r"\b(?:surface\s*area|BET\b|Langmuir)\b",
        "Pore volume": r"\b(?:pore\s*volume|BJH)\b",
        "Pore size": r"\b(?:pore\s*size|pore\s*diameter)\b",
        "Thermal stability": r"\b(?:thermal\s*stability|decomposition|TGA)\b",
    }

    # Structural properties keywords
    STRUCTURAL_PATTERNS = {
        "surface_area": r"(?:surface\s*area|BET|Langmuir|area)\s*(?:=|:)?\s*(\d+\.?\d*)\s*m[2²]\/g",
        "pore_volume": r"(?:pore\s*volume|BJH\s*volume)\s*(?:=|:)?\s*(\d+\.?\d*)\s*cm[3³]\/g",
        "pore_size": r"(?:pore\s*size|pore\s*diameter|BJH\s*diameter)\s*(?:=|:)?\s*(\d+\.?\d*)\s*nm",
        "density": r"(?:density|bulk\s*density)\s*(?:=|:)?\s*(\d+\.?\d*)\s*g\/cm[3³]",
        "particle_size": r"(?:particle\s*size|average\s*size)\s*(?:=|:)?\s*(\d+\.?\d*)\s*(?:nm|μm|micron)",
    }

    # Performance parameters
    PERFORMANCE_PATTERNS = {
        "co2_uptake": r"(?:CO2?\s*uptake|CO2?\s*adsorption\s*capacity)\s*(?:=|:)?\s*(\d+\.?\d*)\s*(?:mmol|mol)\/g",
        "selectivity": r"(?:selectivity|CO2\/N2\s*selectivity)\s*(?:=|:)?\s*(\d+\.?\d*)",
        "adsorption_capacity": r"(?:adsorption\s*capacity|max\s*uptake)\s*(?:=|:)?\s*(\d+\.?\d*)\s*(?:mmol|mol)\/g",
        "desorption_energy": r"(?:desorption\s*energy|heat\s*of\s*desorption)\s*(?:=|:)?\s*(\d+\.?\d*)\s*kJ\/mol",
        "heat_of_adsorption": r"(?:heat\s*of\s*adsorption|isosteric\s*heat)\s*(?:=|:)?\s*(\d+\.?\d*)\s*kJ\/mol",
        "conversion": r"(?:conversion|CO2?\s*conversion)\s*(?:=|:)?\s*(\d+\.?\d*)\s*%",
        "yield": r"(?:yield|product\s*yield)\s*(?:=|:)?\s*(\d+\.?\d*)\s*%",
        "recovery": r"(?:recovery|regeneration)\s*(?:=|:)?\s*(\d+\.?\d*)\s*%",
        "stability_cycles": r"(?:stability|cycles|adsorption/desorption\s*cycles)\s*(?:=|:)?\s*(\d+)\s*(?:cycles|cycle)",
    }

    # Experimental conditions
    CONDITION_PATTERNS = {
        "temperature_K": r"(?:temperature|T\s*=|at|@)\s*(\d+\.?\d*)\s*K",
        "temperature_C": r"(?:temperature|T\s*=|at|@)\s*(\d+\.?\d*)\s*°C",
        "pressure_bar": r"(?:pressure|P\s*=|at)\s*(\d+\.?\d*)\s*(?:bar|Bar)",
        "pressure_atm": r"(?:pressure|P\s*=|at)\s*(\d+\.?\d*)\s*atm",
        "pressure_pa": r"(?:pressure|P\s*=|at)\s*(\d+\.?\d*)\s*(?:Pa|kPa|MPa)",
        "humidity": r"(?:humidity|RH|relative\s*humidity)\s*(?:=|:)?\s*(\d+\.?\d*)\s*%",
        "ph": r"(?:pH|pH\s*=|pH\s*value)\s*(?:=|:)?\s*(\d+\.?\d*)",
        "concentration": r"(?:concentration|conc\.?)\s*(?:=|:)?\s*(\d+\.?\d*)\s*(?:ppm|ppb|mol\/L|M)",
        "contact_time": r"(?:contact\s*time|residence\s*time)\s*(?:=|:)?\s*(\d+\.?\d*)\s*(?:min|hour|h)",
        "flow_rate": r"(?:flow\s*rate|GHSV)\s*(?:=|:)?\s*(\d+\.?\d*)\s*(?:cm3\/min|mL\/min|h-1)",
        "sample_mass": r"(?:sample\s*mass|mass\s*of\s*sample)\s*(?:=|:)?\s*(\d+\.?\d*)\s*(?:mg|g)",
    }

    # Method and evidence keywords
    METHOD_KEYWORDS = {
        "volumetric adsorption": r"\b(?:volumetric|static\s*volumetric)\b",
        "gravimetric adsorption": r"\b(?:gravimetric|microbalance|TGA)\b",
        "breakthrough": r"\b(?:breakthrough|BTF)\b",
        "isotherm": r"\b(?:isotherm|adsorption\s*isotherm)\b",
        "GCMC": r"\b(?:GCMC|Grand\s*Canonical)\b",
        "NPT": r"\b(?:NPT|isothermal.?isobaric)\b",
        "DFT": r"\b(?:DFT|density\s*functional)\b",
        "MD": r"\b(?:MD\b|molecular\s*dynamics)\b",
        "calorimetry": r"\b(?:calorimetry|DSC|differential\s*scanning)\b",
    }

    def __init__(self):
        """Initialize enricher."""
        self.logger = logger
        
        # Pre-compile all regex patterns for better performance
        self.compiled_material_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.MATERIAL_PATTERNS.items()
        }
        self.compiled_measurement_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.MEASUREMENT_PATTERNS.items()
        }
        self.compiled_structural_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.STRUCTURAL_PATTERNS.items()
        }
        self.compiled_performance_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PERFORMANCE_PATTERNS.items()
        }
        self.compiled_condition_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.CONDITION_PATTERNS.items()
        }
        self.compiled_method_keywords = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.METHOD_KEYWORDS.items()
        }

    def enrich(
        self,
        text: str,
        doc_id: int,
        chunk_id: int,
        filename: str,
        page_number: int,
        section: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Enrich chunk with structured chemistry metadata.

        Args:
            text: Chunk text
            doc_id: Document ID
            chunk_id: Chunk ID
            filename: Source filename
            page_number: Page number
            section: Document section

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            # Document identity
            "document_id": f"paper_{doc_id:03d}",
            "filename": filename,
            "section": section,
            "page": page_number,
        }

        # Extract all metadata categories
        metadata.update(self._extract_material_identity(text))
        metadata.update(self._extract_structural_properties(text))
        metadata.update(self._extract_performance_parameters(text))
        metadata.update(self._extract_conditions(text))
        metadata.update(self._extract_method_evidence(text))

        # Infer source type
        metadata["source_type"] = self._infer_source_type(text)
        metadata["gas"] = self._extract_gas_type(text)

        return metadata

    def _extract_material_identity(self, text: str) -> Dict[str, Any]:
        """Extract material name, class, formula, composition."""
        metadata = {}

        # Material class (use pre-compiled patterns)
        for mat_class, pattern in self.compiled_material_patterns.items():
            if pattern.search(text):
                metadata["material_class"] = mat_class
                break
        else:
            metadata["material_class"] = None

        # Material name (look for common patterns like "UiO-66-NH2", "MIL-101")
        name_match = re.search(
            r"\b(?:material|compound|sorbent|adsorbent)\s+(?:named\s+)?([A-Z][A-Za-z0-9\-\.]+)\b",
            text,
        )
        metadata["material_name"] = name_match.group(1) if name_match else None

        # Chemical formula
        formula_match = re.search(
            r"\b([A-Z][a-z]?(?:\d+)?(?:\([A-Za-z0-9\-]+\))?(?:\d+)?)\b",
            text,
        )
        metadata["chemical_formula"] = formula_match.group(1) if formula_match else None

        # Metal center (common in MOFs)
        metal_match = re.search(
            r"\b(?:metal|center|active\s+site)\s*(?:=|:)?\s*([A-Z][a-z]?)\b",
            text,
        )
        metadata["metal_center"] = metal_match.group(1) if metal_match else None

        # Functional groups (use pre-compiled patterns)
        functional_groups = []
        if re.search(r"\bNH2\b", text):
            functional_groups.append("NH2")
        if re.search(r"\bCOOH\b", text):
            functional_groups.append("COOH")
        if re.search(r"\bOH\b", text):
            functional_groups.append("OH")
        if re.search(r"\bF\b", text):
            functional_groups.append("F")
        metadata["functional_groups"] = ",".join(functional_groups) if functional_groups else None

        # Dopants
        dopants = []
        if re.search(r"\bdoped\b|doping", text, re.IGNORECASE):
            dopants.append("doped")
        if re.search(r"\bN-doped\b", text):
            dopants.append("N-doped")
        if re.search(r"\bS-doped\b", text):
            dopants.append("S-doped")
        metadata["dopants"] = ",".join(dopants) if dopants else None

        return metadata

    def _extract_structural_properties(self, text: str) -> Dict[str, Any]:
        """Extract surface area, pore volume, pore size, density, etc."""
        metadata = {}

        for prop, pattern in self.compiled_structural_patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    value = float(match.group(1))
                    if prop == "surface_area":
                        metadata["surface_area_m2_g"] = value
                    elif prop == "pore_volume":
                        metadata["pore_volume_cm3_g"] = value
                    elif prop == "pore_size":
                        metadata["pore_size_nm"] = value
                    elif prop == "density":
                        metadata["density_g_cm3"] = value
                    elif prop == "particle_size":
                        metadata["particle_size_nm"] = value
                except (ValueError, IndexError):
                    pass

        # Crystal structure keywords
        if re.search(r"\bcubic\b", text, re.IGNORECASE):
            metadata["crystal_structure"] = "cubic"
        elif re.search(r"\btetragonal\b", text, re.IGNORECASE):
            metadata["crystal_structure"] = "tetragonal"
        elif re.search(r"\bhexagonal\b", text, re.IGNORECASE):
            metadata["crystal_structure"] = "hexagonal"
        else:
            metadata["crystal_structure"] = None

        # Topology
        topology_match = re.search(r"\b(?:topology|topological)\s*(?:=|:)?\s*(\w+)\b", text)
        if topology_match:
            metadata["topology"] = topology_match.group(1)

        return metadata

    def _extract_performance_parameters(self, text: str) -> Dict[str, Any]:
        """Extract CO2 uptake, selectivity, stability, etc."""
        metadata = {}

        for param, pattern in self.compiled_performance_patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    value = float(match.group(1))
                    if param == "co2_uptake":
                        metadata["co2_uptake_mmol_g"] = value
                    elif param == "stability_cycles":
                        metadata["stability_cycles"] = int(value)
                    else:
                        if param == "heat_of_adsorption":
                            metadata["heat_of_adsorption_kJ_mol"] = value
                        elif param == "desorption_energy":
                            metadata["desorption_energy_kJ_mol"] = value
                        elif param == "conversion":
                            metadata["conversion_percent"] = value
                        elif param == "yield":
                            metadata["yield_percent"] = value
                        elif param == "recovery":
                            metadata["recovery_percent"] = value
                        else:
                            metadata[param] = value
                except (ValueError, IndexError):
                    pass

        return metadata

    def _extract_conditions(self, text: str) -> Dict[str, Any]:
        """Extract experimental conditions."""
        metadata = {}

        # Temperature (prefer Kelvin, convert from Celsius if needed)
        temp_k = self.compiled_condition_patterns["temperature_K"].search(text)
        if temp_k:
            metadata["temperature_K"] = float(temp_k.group(1))
        else:
            temp_c = self.compiled_condition_patterns["temperature_C"].search(text)
            if temp_c:
                metadata["temperature_K"] = float(temp_c.group(1)) + 273.15

        # Pressure (prefer bar, convert if needed)
        pres_bar = self.compiled_condition_patterns["pressure_bar"].search(text)
        if pres_bar:
            metadata["pressure_bar"] = float(pres_bar.group(1))
        else:
            pres_atm = self.compiled_condition_patterns["pressure_atm"].search(text)
            if pres_atm:
                metadata["pressure_bar"] = float(pres_atm.group(1)) * 1.01325

        # Other conditions
        for cond in ["humidity", "ph", "concentration", "contact_time", "flow_rate", "sample_mass"]:
            if cond in self.compiled_condition_patterns:
                match = self.compiled_condition_patterns[cond].search(text)
                if match:
                    try:
                        metadata[cond] = float(match.group(1))
                    except (ValueError, IndexError):
                        pass

        # Gas composition
        if re.search(r"\b(?:pure|100%)\s+CO2\b", text, re.IGNORECASE):
            metadata["gas_composition"] = "pure CO2"
        elif re.search(r"\bCO2/N2\b", text):
            metadata["gas_composition"] = "CO2/N2 mixture"

        return metadata

    def _extract_method_evidence(self, text: str) -> Dict[str, Any]:
        """Extract measurement method, simulation method, instrument, error info."""
        metadata = {}

        # Measurement method
        for method, pattern in self.compiled_method_keywords.items():
            if pattern.search(text):
                if "adsorption" in method.lower():
                    metadata["measurement_method"] = method
                else:
                    metadata["simulation_method"] = method

        # Instrument names
        instruments = []
        if re.search(r"\bBELSORP\b", text):
            instruments.append("BELSOPR")
        if re.search(r"\bAUTOSORB\b", text):
            instruments.append("AUTOSORB")
        if re.search(r"\bAutosorb", text):
            instruments.append("Autosorb")
        if re.search(r"\bTGA\b", text):
            instruments.append("TGA")
        if instruments:
            metadata["instrument"] = ",".join(instruments)

        # Uncertainty/error
        uncertainty = re.search(
            r"(?:uncertainty|error|error\s+range)\s*(?:=|:)?\s*([±±]\s*)?(\d+\.?\d*)\s*%",
            text,
        )
        if uncertainty:
            metadata["uncertainty_percent"] = float(uncertainty.group(2))

        # Replicate count
        replicate = re.search(
            r"(?:replicate|repeated|triplicate|duplicate)\s*(?:=|:)?\s*(\d+)",
            text,
        )
        if replicate:
            metadata["replicate_count"] = int(replicate.group(1))

        # Table/figure reference
        table_ref = re.search(r"(?:Table|Fig\.?|Figure)\s*(\d+[a-z]?)", text)
        if table_ref:
            metadata["table_or_figure_reference"] = table_ref.group(0)

        return metadata

    def _infer_source_type(self, text: str) -> str:
        """Infer if source is experimental or simulated/computational."""
        computational_keywords = [
            "simulation",
            "GCMC",
            "molecular dynamics",
            "DFT",
            "computational",
            "modeled",
            "calculated",
        ]
        experimental_keywords = [
            "experimental",
            "measured",
            "volumetric",
            "gravimetric",
            "observed",
        ]

        experimental_score = sum(
            1 for kw in experimental_keywords if re.search(kw, text, re.IGNORECASE)
        )
        computational_score = sum(
            1 for kw in computational_keywords if re.search(kw, text, re.IGNORECASE)
        )

        if computational_score > experimental_score:
            return "computational"
        elif experimental_score > 0:
            return "experimental"
        else:
            return "unknown"

    def _extract_gas_type(self, text: str) -> Optional[str]:
        """Extract primary gas type from text."""
        gases = {
            "CO2": r"\bCO2?\b",
            "N2": r"\bN2\b",
            "CH4": r"\bCH4\b",
            "O2": r"\bO2\b",
            "H2": r"\bH2\b",
            "He": r"\bHe\b",
            "Ar": r"\bAr\b",
            "C2H6": r"\bC2H6\b",
            "C3H8": r"\bC3H8\b",
        }

        for gas, pattern in gases.items():
            if re.search(pattern, text):
                return gas

        return None


if __name__ == "__main__":
    # Quick test
    enricher = MetadataEnricher()
    test_text = """
    The MOF UiO-66-NH2 (Zr6O4(OH)4(BDC-NH2)6) with a surface area of 1120 m2/g 
    and pore volume of 0.48 cm3/g achieved a CO2 uptake of 4.2 mmol/g 
    at 298 K and 1 bar. The heat of adsorption was 35 kJ/mol. 
    Experimental measurements were performed using volumetric adsorption.
    """
    result = enricher.enrich(test_text, doc_id=18, chunk_id=4, filename="test.pdf", page_number=8)
    import json
    print(json.dumps(result, indent=2, default=str))
