"""Pure-Python tests for component-name normalization and grouping."""

import pytest

from kgrag.extract.schema import Mention
from kgrag.resolve.components import component_key, group_components, split_version


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Widget Maker 2", "WidgetMaker2"),
        ("widget-maker", "Widget Maker"),
        ("widget_maker", "WidgetMaker"),
        ("Kafka Widget", "Widget"),
        ("kafka-widget", "Widget"),
        ("Apache Widget", "Widget"),
        ("the widget", "Widget"),
        ("widgets", "Widget"),
        ("FooManager.java", "FooManager"),
        ("org.example.FooManager", "FooManager"),
    ],
)
def test_component_key_equal(a, b):
    assert component_key(a) == component_key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("FooExtension", "FooExtensionContext"),
        ("AsyncFooClient", "LegacyFooClient"),
        ("KafkaWidget", "Widget"),  # no separator: a class name, not a prefix
        ("Widget 2", "Widget"),  # version stripping is a separate, guarded step
    ],
)
def test_component_key_distinct(a, b):
    assert component_key(a) != component_key(b)


def test_component_key_keeps_class_name_plurals():
    assert component_key("FooSerdes") != component_key("FooSerde")


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Widget 12", ("Widget", "12")),
        ("Widget 3.4", ("Widget", "3.4")),
        ("Widget v12", ("Widget", "v12")),
        ("Widget 2", ("Widget 2", None)),  # a single digit is usually part of the name
        ("Widget2", ("Widget2", None)),
        ("Widget", ("Widget", None)),
    ],
)
def test_split_version(surface, expected):
    assert split_version(surface) == expected


def _m(i, surface):
    return Mention(mention_id=f"e:m{i}", surface=surface, type="Component", email_id="e", context="")


def test_group_components_merges_versions_only_when_base_exists():
    ms = [_m(1, "Widget"), _m(2, "Widget 12"), _m(3, "Gadget 12"), _m(4, "Gadget 13")]
    groups = {frozenset(m.mention_id for m in g) for _, g in group_components(ms)}
    assert groups == {frozenset({"e:m1", "e:m2"}), frozenset({"e:m3"}), frozenset({"e:m4"})}


def test_group_components_canonical_is_most_common_then_longest():
    ms = [_m(1, "widget"), _m(2, "Widget"), _m(3, "Widget"), _m(4, "Kafka Widget")]
    [(name, group)] = group_components(ms)
    assert name == "Widget" and len(group) == 4


def test_dotted_version_prefers_product_named_with_major_version():
    ms = [_m(1, "Widget"), _m(2, "Widget2"), _m(3, "Widget 2.0"), _m(4, "Widget 3.4")]
    groups = {frozenset(m.mention_id for m in g) for _, g in group_components(ms)}
    assert groups == {frozenset({"e:m1", "e:m4"}), frozenset({"e:m2", "e:m3"})}
