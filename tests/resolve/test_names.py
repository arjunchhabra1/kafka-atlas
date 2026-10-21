"""Pure-Python tests for the person-name helpers used by UserResolver."""

import pytest

from kgrag.resolve.names import (
    address_local_part,
    clean_display_name,
    first_names_compatible,
    fold,
    full_key,
    is_greeting,
    is_relay_name,
    is_surname_typo,
    parse_name,
)


def test_fold_casefolds_and_strips_diacritics():
    assert fold("José GARCÍA") == "jose garcia"
    assert fold("Strauß") == "strauss"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Jane Doe via dev", "Jane Doe"),
        ("Doe, Jane", "Jane Doe"),
        ("Warren, Jr", "Warren"),
        ("Jane Doe, Jr", "Jane Doe"),
        ("  Jane   Doe  ", "Jane Doe"),
        ('"Jane Doe"', "Jane Doe"),
        ("Jane Doe (Jira)", "Jane Doe"),
        ("", ""),
    ],
)
def test_clean_display_name(raw, expected):
    assert clean_display_name(raw) == expected


def test_is_relay_name():
    assert is_relay_name("Jane Doe via dev")
    assert not is_relay_name("Jane Doe")


def test_parse_name_full_name():
    n = parse_name("Jane Q. Doe")
    assert (n.first, n.last, n.is_full, n.last_is_initial) == ("jane", "doe", True, False)


def test_parse_name_hyphenated_first_name_is_one_token():
    n = parse_name("Mei-Lin Wu")
    assert (n.first, n.last) == ("meilin", "wu")
    assert parse_name("Mei-Lin").first == "meilin"
    assert parse_name("Mei-Lin").last is None


def test_parse_name_first_plus_initial():
    for raw in ("Jane D.", "Jane D"):
        n = parse_name(raw)
        assert (n.first, n.last, n.is_full, n.last_is_initial) == ("jane", "d", False, True)


def test_parse_name_strips_greetings_and_other():
    assert parse_name("Hi Jane").tokens == ("jane",)
    assert parse_name("Other Jane").tokens == ("jane",)
    assert parse_name("Hello").tokens == ("hello",)  # never strip down to nothing


def test_parse_name_first_only():
    n = parse_name("Jane")
    assert (n.first, n.last, n.is_full) == ("jane", None, False)


def test_full_key_uses_formal_first_name_and_drops_middle():
    assert full_key(parse_name("Bob Q. Smith")) == full_key(parse_name("Robert Smith")) == "robert smith"
    assert full_key(parse_name("Jane")) is None
    assert full_key(parse_name("Jane D.")) is None


def test_first_names_compatible_is_asymmetric():
    assert first_names_compatible("jane", "jane")
    assert first_names_compatible("chris", "christopher")  # short form names the formal one
    assert first_names_compatible("bob", "rob")  # two short forms of the same name
    assert not first_names_compatible("christopher", "chris")  # formal never means a short form
    assert not first_names_compatible("jane", "john")


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("Hi Jane,", True),
        ("hello Jane, thanks", True),
        ("Hey Jane", True),
        ("Dear Jane,", True),
        ("Thanks, Jane", False),
        ("Jane", False),
        ("as Jane said, hi", False),
    ],
)
def test_is_greeting(context, expected):
    assert is_greeting(context) is expected


def test_address_local_part():
    assert address_local_part("Jane.Doe@Example.org") == "janedoe"
    assert address_local_part("not an address") is None


def test_parse_name_splits_on_dots():
    assert parse_name("Jane Q.Doe").last == "doe"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("cadenza", "cadenzo", True), ("cadenza", "cadenzza", True), ("cadenza", "cadenza", True),
     ("cadenza", "kadenzo", False), ("lee", "lea", False)],
)
def test_is_surname_typo(a, b, expected):
    assert is_surname_typo(a, b) is expected
