import pytest

from kgrag.extract.regex_ids import IdMention, extract_ids


def ids(text: str, kind: str) -> list[str]:
    return [m.id for m in extract_ids(text) if m.kind == kind]


@pytest.mark.parametrize(
    "text",
    ["KIP-848", "kip-848", "KIP 848", "KIP848", "see KIP-848: the protocol", "(KIP-848)"],
)
def test_kip_variants(text):
    assert ids(text, "KIP") == ["KIP-848"]


@pytest.mark.parametrize("text", ["SKIP-1", "KIPS-1", "KIP-", "my KIPX 5", "KIP-12abc"])
def test_kip_word_boundaries(text):
    assert ids(text, "KIP") == []


def test_ticket():
    assert ids("fixed in KAFKA-1234 and kafka-99", "JiraTicket") == ["KAFKA-1234", "KAFKA-99"]


def test_ticket_in_url_is_kept():
    text = "https://issues.apache.org/jira/browse/KAFKA-1234"
    [m] = extract_ids(text)
    assert (m.kind, m.id) == ("JiraTicket", "KAFKA-1234")
    assert text[slice(*m.span)] == "KAFKA-1234"


def test_ticket_word_boundaries():
    assert ids("XKAFKA-12 KAFKA-12x KAFKA-", "JiraTicket") == []


def test_release_needs_context():
    assert ids("bump to 3.7.0", "Release") == ["3.7.0"]
    assert ids("ratio 3.5 seconds", "Release") == []


def test_no_false_kafka_word():
    assert extract_ids("Kafka 3") == []


def test_release_xy_with_keyword_before_and_after():
    assert ids("in release 3.7 we", "Release") == ["3.7"]
    assert ids("the 3.7 release is out", "Release") == ["3.7"]
    assert ids("on the 3.6 branch", "Release") == ["3.6"]


def test_release_xy_and_xyz_stay_distinct():
    assert ids("version 3.7 and version 3.7.0", "Release") == ["3.7", "3.7.0"]


@pytest.mark.parametrize("text", ["host 10.0.0.1", "ip 1.2.3.4", "version 3.7.0.1", "release 1.2.3.4.5"])
def test_release_not_inside_longer_dotted_number(text):
    assert ids(text, "Release") == []


def test_release_two_digit_major_out_of_scope():
    assert ids("JDK 17.0.2 release", "Release") == []


@pytest.mark.parametrize("text", ["3.7.0-rc1", "3.7.0 RC2", "3.7.0-SNAPSHOT"])
def test_release_suffixes(text):
    [m] = extract_ids(text)
    assert (m.kind, m.id) == ("Release", "3.7.0")
    assert text[slice(*m.span)] == "3.7.0"


def test_release_xy_rc_suffix_gives_context():
    assert ids("cutting 3.7-rc1 today", "Release") == ["3.7"]


def test_release_keyword_distance_boundary():
    assert ids("release one two three 3.7", "Release") == ["3.7"]  # 3 words between
    assert ids("release one two three four 3.7", "Release") == []  # 4 words between
    assert ids("3.7 one two three release", "Release") == ["3.7"]
    assert ids("3.7 one two three four release", "Release") == []


def test_release_keywords_case_insensitive_and_punctuated():
    assert ids("VERSION: 3.7", "Release") == ["3.7"]
    assert ids("RC 3.7", "Release") == ["3.7"]


def test_release_not_followed_by_more_digits():
    assert ids("release 3.123", "Release") == []


def test_ordering_spans_and_repeats():
    text = "KAFKA-1 then KIP-2 then KAFKA-1 in 3.7.0"
    got = extract_ids(text)
    assert [m.id for m in got] == ["KAFKA-1", "KIP-2", "KAFKA-1", "3.7.0"]
    assert got[0].span == (0, 7) and got[2].span == (24, 31)
    assert all(isinstance(m, IdMention) for m in got)
    for m in got:
        assert m.span[0] < m.span[1]
