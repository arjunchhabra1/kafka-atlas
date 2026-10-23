"""Pure-Python tests for the person-name helpers used by UserResolver."""

import pytest

from kgrag.resolve.names import (
    address_local_part,
    clean_display_name,
    first_names_compatible,
    fold,
    full_key,
    is_relay_name,
    name_role,
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
    ("context", "surface"),
    [
        ("Jane", "Jane"),
        ("Thanks,\nJane", "Jane"),
        ("Thanks, Jane", "Jane"),
        ("Thanks!  Jane", "Jane"),
        ("Best, Jane", "Jane"),
        ("Kind regards,\nJane", "Jane"),
        ("Thanks again, Jane", "Jane"),
        ("Thanks and best regards, Jane", "Jane"),
        ("TIA Jane", "Jane"),
        ("HTH Jane", "Jane"),
        ("-- Jane", "Jane"),
        ("-Jane", "-Jane"),
        ("Cheers\nJane D", "Jane D"),
        ("Thanks so much! Jane", "Jane"),  # a closing sentence, then the name
        ("Thanks, everyone. Jane.", "Jane"),
        ("Jane\non behalf of the PMC", "Jane"),
        ("I'm Jane, a developer at Example", "Jane"),  # self-introduction
        ("Hope this helps, Jane", "Jane"),  # last words of the message
        ("Thanks a lot!   -Jane", "Jane"),
        ("Best, Jane & Bo", "Jane"),  # co-signed
        ("Jane and the Widget team", "Jane"),
    ],
)
def test_name_role_self(context, surface):
    assert name_role(context, surface) == "self"


def test_name_role_self_allows_the_senders_surname_after_the_name():
    assert name_role("Best,\nJane Doe", "Jane", surnames={"doe"}) == "self"
    assert name_role("Best,\nJane Doe", "Jane") == "other"


@pytest.mark.parametrize(
    ("context", "surface"),
    [
        ("Thanks, Jane. This is a nice addition!", "Jane"),  # thanks + more sentence
        ("Thank you Jane for the KIP", "Jane"),
        ("based on Jane's comment", "Jane"),  # possessive
        ("@Jane: can you prepare a KIP?", "Jane"),  # @mention
        ("@Jane", "Jane"),
        ("Jane: can you take a look?", "Jane"),  # vocative
        ("Jane, could you check?", "Jane"),
        ("as Jane said, we should wait", "Jane"),  # as X said
        ("as suggested by Jane", "Jane"),  # name at the end, but not after a closing word
        ("Thanks Jane again", "Jane"),
        ("thanks to Jane", "Jane"),
        ("4 binding votes (Bo, Jane and Cy)", "Jane"),
    ],
)
def test_name_role_other(context, surface):
    assert name_role(context, surface) == "other"


@pytest.mark.parametrize("context", ["Hi Jane,", "hello Jane, thanks", "Dear Jane", "Welcome Jane"])
def test_name_role_greeting(context):
    assert name_role(context, "Jane") == "greeting"


def test_name_role_absent_when_the_sentence_does_not_contain_the_name():
    assert name_role("I can update the KIP.", "Jane") == "absent"


def test_parse_name_lists_of_people_are_not_a_full_name():
    assert not parse_name("Ann, Bo and Cy").is_full
    assert not parse_name("Ann Bo Cy Dee Eve").is_full


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


def test_name_role_ignores_greeting_inside_the_surface():
    assert name_role("Hi Jane,", "Hi Jane") == "greeting"
    assert name_role("Thanks,\nJane", "Thanks Jane") == "self"


def test_name_role_matches_whole_words_only():
    assert name_role("Thanks, Joanne", "Ann") == "absent"
