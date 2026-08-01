"""Tests for the shared truncate/seal helpers in ``_action_binding``.

Every framework adapter is supposed to route its input-truncation and
metadata-sealing through these three functions instead of reimplementing
them locally — see ``test_action_binding_contract.py`` for the structural
check that enforces that.
"""

from __future__ import annotations

import json

import pytest

from datetime import datetime, timezone

from execlave.errors import MetadataContractError
from execlave.integrations._action_binding import (
    truncate_for_classifier,
    seal_for_metadata,
    seal_metadata_entry,
    seal_metadata,
    seal_full_request,
    enforce_policy_bound,
    SealedMetadata,
)


class TestTruncateForClassifier:
    def test_returns_none_for_none(self) -> None:
        assert truncate_for_classifier(None) is None

    def test_passes_a_string_through_untouched_below_the_limit(self) -> None:
        assert truncate_for_classifier("hello") == "hello"

    def test_truncates_at_the_given_limit(self) -> None:
        big = "x" * 5000
        assert len(truncate_for_classifier(big)) == 4000
        assert len(truncate_for_classifier(big, 10)) == 10

    def test_stringifies_a_non_string_value(self) -> None:
        assert truncate_for_classifier({"q": "x"}) == "{'q': 'x'}"

    def test_does_not_raise_for_a_circular_reference(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        result = truncate_for_classifier(circular)
        assert isinstance(result, str)


class TestSealForMetadata:
    def test_returns_none_for_none(self) -> None:
        assert seal_for_metadata(None) is None

    def test_round_trips_a_plain_value_unchanged(self) -> None:
        value = {"a": 1, "b": ["x", "y"], "c": {"nested": True}}
        assert seal_for_metadata(value) == value

    def test_never_raises_for_a_circular_reference(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        result = seal_for_metadata(circular)
        assert result == {"self": "[unserializable:circular]"}

    def test_fallback_is_json_serializable(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        json.dumps(seal_for_metadata(circular))

    def test_isolates_a_single_bad_field_instead_of_collapsing_the_whole_value(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        value = {"good": "kept", "model": "gpt-4o", "bad": circular}
        assert seal_for_metadata(value) == {
            "good": "kept",
            "model": "gpt-4o",
            "bad": {"self": "[unserializable:circular]"},
        }

    def test_marks_non_serializable_fields_explicitly_instead_of_raising(self) -> None:
        value = {"cb": lambda: None, "raw": b"bytes", "tag": {1, 2}, "fine": "ok"}
        result = seal_for_metadata(value)
        assert result["cb"] == "[unserializable:callable]"
        assert result["raw"] == "[unserializable:bytes:5]"
        assert result["tag"] == "[unserializable:set]"
        assert result["fine"] == "ok"

    def test_serializes_datetime_to_isoformat(self) -> None:
        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert seal_for_metadata({"when": when}) == {"when": when.isoformat()}

    def test_does_not_falsely_flag_a_shared_sibling_reference_as_circular(self) -> None:
        shared = {"q": "x"}
        assert seal_for_metadata({"a": shared, "b": shared}) == {
            "a": {"q": "x"},
            "b": {"q": "x"},
        }


class TestSealFullRequest:
    def test_seals_every_field_by_default(self) -> None:
        params = {"model": "gpt-4o", "messages": [], "tools": [{"type": "function"}], "temperature": 0.2}
        assert seal_full_request(params) == params

    def test_strips_only_named_exclusions(self) -> None:
        params = {"model": "gpt-4o", "signal": lambda: None, "temperature": 0.2}
        assert seal_full_request(params, exclude=("signal",)) == {
            "model": "gpt-4o",
            "temperature": 0.2,
        }

    def test_returns_empty_sealed_metadata_for_a_non_dict_request(self) -> None:
        assert seal_full_request(None) == {}

    def test_returns_a_sealed_metadata_instance(self) -> None:
        assert isinstance(seal_full_request({"model": "gpt-4o"}), SealedMetadata)


class TestSealMetadataEntry:
    def test_returns_none_when_there_is_nothing_to_seal(self) -> None:
        assert seal_metadata_entry("key", None) is None

    def test_wraps_a_sealed_value_under_the_given_key(self) -> None:
        assert seal_metadata_entry("toolArguments", {"q": "x"}) == {
            "toolArguments": {"q": "x"},
        }

    def test_is_json_serializable_even_for_a_circular_input(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        entry = seal_metadata_entry("toolArguments", circular)
        json.dumps(entry)

    def test_returns_a_sealed_metadata_instance(self) -> None:
        entry = seal_metadata_entry("toolArguments", {"q": "x"})
        assert isinstance(entry, SealedMetadata)


class TestSealMetadata:
    def test_seals_a_hand_built_object_combining_plain_and_structured_fields(self) -> None:
        sealed = seal_metadata({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        assert sealed == {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        assert isinstance(sealed, SealedMetadata)

    def test_falls_back_to_a_safe_marker_for_a_circular_reference(self) -> None:
        circular: dict = {}
        circular["self"] = circular
        sealed = seal_metadata({"payload": circular})
        json.dumps(sealed)


class TestEnforcePolicyBound:
    def _make_exe(self):
        class _Exe:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def enforce_policy(self, agent_id, input, *, tools=None, metadata=None, **kwargs):
                self.calls.append(
                    {"agent_id": agent_id, "input": input, "tools": tools, "metadata": metadata}
                )
                return {"allowed": True}

        return _Exe()

    def test_forwards_a_seal_metadata_entry_result(self) -> None:
        exe = self._make_exe()
        metadata = seal_metadata_entry("toolArguments", {"q": "x"})
        enforce_policy_bound(exe, "a1", "tool:search", metadata=metadata)
        assert exe.calls == [
            {"agent_id": "a1", "input": "tool:search", "tools": None, "metadata": metadata}
        ]

    def test_forwards_a_seal_metadata_result(self) -> None:
        exe = self._make_exe()
        metadata = seal_metadata({"model": "gpt-4o", "messages": []})
        enforce_policy_bound(exe, "a1", "hi", metadata=metadata)
        assert exe.calls[0]["metadata"] == metadata

    def test_allows_none_metadata_through_untouched(self) -> None:
        exe = self._make_exe()
        enforce_policy_bound(exe, "a1", "hi")
        assert exe.calls == [{"agent_id": "a1", "input": "hi", "tools": None, "metadata": None}]

    def test_raises_metadata_contract_error_for_a_plain_dict_without_calling_enforce_policy(
        self,
    ) -> None:
        exe = self._make_exe()
        with pytest.raises(MetadataContractError):
            enforce_policy_bound(exe, "a1", "hi", metadata={"q": "x"})
        assert exe.calls == []
