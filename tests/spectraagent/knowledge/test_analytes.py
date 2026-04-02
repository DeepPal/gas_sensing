"""
Unit tests for spectraagent.knowledge.analytes.
"""
from __future__ import annotations

import pytest

from spectraagent.knowledge.analytes import (
    ANALYTE_REGISTRY,
    AnalyteProperties,
    format_analyte_chemistry_brief,
    list_known_analytes,
    lookup_analyte,
)


# ---------------------------------------------------------------------------
# AnalyteProperties dataclass
# ---------------------------------------------------------------------------

class TestAnalyteProperties:
    def test_required_fields_only(self):
        ap = AnalyteProperties(name="TestGas", cas_number="0-00-0")
        assert ap.name == "TestGas"
        assert ap.cas_number == "0-00-0"

    def test_optional_fields_default(self):
        ap = AnalyteProperties(name="X", cas_number="0-00-0")
        assert ap.functional_group == "unknown"
        assert ap.osha_pel_ppm is None
        assert ap.idlh_ppm is None
        assert ap.known_interferents == []

    def test_frozen_dataclass(self):
        ap = AnalyteProperties(name="X", cas_number="0-00-0")
        with pytest.raises((AttributeError, TypeError)):
            ap.name = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ANALYTE_REGISTRY contents
# ---------------------------------------------------------------------------

class TestAnalyteRegistry:
    def test_registry_not_empty(self):
        assert len(ANALYTE_REGISTRY) > 5

    def test_ethanol_in_registry(self):
        assert "ethanol" in ANALYTE_REGISTRY

    def test_methanol_in_registry(self):
        assert "methanol" in ANALYTE_REGISTRY

    def test_isopropanol_in_registry(self):
        assert "isopropanol" in ANALYTE_REGISTRY

    def test_alias_ethyl_alcohol(self):
        assert "ethyl_alcohol" in ANALYTE_REGISTRY
        assert ANALYTE_REGISTRY["ethyl_alcohol"] is ANALYTE_REGISTRY["ethanol"]

    def test_alias_ipa(self):
        assert "ipa" in ANALYTE_REGISTRY
        assert ANALYTE_REGISTRY["ipa"] is ANALYTE_REGISTRY["isopropanol"]

    def test_all_entries_are_analyte_properties(self):
        for key, val in ANALYTE_REGISTRY.items():
            assert isinstance(val, AnalyteProperties), f"{key} is not AnalyteProperties"

    def test_ethanol_has_formula(self):
        eth = ANALYTE_REGISTRY["ethanol"]
        assert eth.formula != ""
        assert eth.functional_group == "alcohol"


# ---------------------------------------------------------------------------
# lookup_analyte
# ---------------------------------------------------------------------------

class TestLookupAnalyte:
    def test_exact_match(self):
        result = lookup_analyte("ethanol")
        assert result is not None
        assert result.name.lower() == "ethanol"

    def test_case_insensitive_upper(self):
        result = lookup_analyte("ETHANOL")
        assert result is not None

    def test_case_insensitive_mixed(self):
        result = lookup_analyte("Isopropanol")
        assert result is not None

    def test_alias_resolution(self):
        alias = lookup_analyte("IPA")
        direct = lookup_analyte("Isopropanol")
        assert alias is not None
        assert direct is not None
        assert alias.name == direct.name

    def test_unknown_returns_none(self):
        assert lookup_analyte("xenon_fluoride") is None

    def test_empty_string_returns_none(self):
        assert lookup_analyte("") is None

    def test_whitespace_stripped(self):
        result = lookup_analyte("  ethanol  ")
        assert result is not None


# ---------------------------------------------------------------------------
# list_known_analytes
# ---------------------------------------------------------------------------

class TestListKnownAnalytes:
    def test_returns_list(self):
        result = list_known_analytes()
        assert isinstance(result, list)

    def test_no_duplicates(self):
        result = list_known_analytes()
        assert len(result) == len(set(result))

    def test_sorted(self):
        result = list_known_analytes()
        assert result == sorted(result)

    def test_contains_canonical_names_not_aliases(self):
        result = list_known_analytes()
        # Aliases like "ethyl_alcohol" should not appear; only "Ethanol"
        assert "Ethanol" in result or "ethanol" in result
        assert "ethyl_alcohol" not in result


# ---------------------------------------------------------------------------
# format_analyte_chemistry_brief
# ---------------------------------------------------------------------------

class TestFormatChemistryBrief:
    def test_contains_name(self):
        eth = ANALYTE_REGISTRY["ethanol"]
        brief = format_analyte_chemistry_brief(eth)
        assert "Ethanol" in brief

    def test_contains_cas(self):
        eth = ANALYTE_REGISTRY["ethanol"]
        brief = format_analyte_chemistry_brief(eth)
        assert eth.cas_number in brief

    def test_contains_formula(self):
        eth = ANALYTE_REGISTRY["ethanol"]
        brief = format_analyte_chemistry_brief(eth)
        assert eth.formula in brief

    def test_returns_string(self):
        eth = ANALYTE_REGISTRY["ethanol"]
        assert isinstance(format_analyte_chemistry_brief(eth), str)
