"""Pure-Python scenario tests for person resolution (invented names, no database)."""

import random

from kgrag.extract.schema import AnchorHints, Mention
from kgrag.resolve.people import resolve_people


def header(email, name, addr):
    return Mention(mention_id=f"{email}:h0", surface=name, type="Person", email_id=email,
                   context=f"{name} {addr}")


def llm(email, n, surface, context):
    return Mention(mention_id=f"{email}:m{n}", surface=surface, type="Person", email_id=email,
                   context=context)


def jira(n, name):
    return Mention(mention_id=f"jira:{n}:a0", surface=name, type="Person", email_id=f"jira-{n}",
                   context=name)


def hints_for(mentions):
    pairs = sorted({(m.surface, m.context.rsplit(" ", 1)[-1]) for m in mentions
                    if m.mention_id.endswith(":h0")})
    actors = sorted({m.surface for m in mentions if m.mention_id.startswith("jira:")})
    return AnchorHints(sender_pairs=pairs, jira_actors=actors)


def clusters(mentions):
    """{frozenset of mention ids: canonical name}."""
    return {frozenset(m.mention_id for m in ms): name
            for name, ms in resolve_people(mentions, hints_for(mentions))}


def entity_of(mentions, mid):
    for ids, name in clusters(mentions).items():
        if mid in ids:
            return ids, name
    raise KeyError(mid)


def test_same_address_and_same_full_name_merge():
    ms = [
        header("a1", "Mei-Lin W.", "mlw@home.example"),
        header("a2", "Mei-Lin Wu", "mlw@home.example"),
        header("a3", "Mei-Lin Wu", "meilin@work.example"),
        jira(1, "Mei-Lin Wu"),
    ]
    ids, name = entity_of(ms, "a1:h0")
    assert ids == {"a1:h0", "a2:h0", "a3:h0", "jira:1:a0"}
    assert name == "Mei-Lin Wu"


def test_relay_address_does_not_merge_senders():
    ms = [header("a1", "Ann Lee via dev", "dev@list.example"),
          header("a2", "Bo Chan via dev", "dev@list.example")]
    assert len(clusters(ms)) == 2
    assert entity_of(ms, "a1:h0")[1] == "Ann Lee"


def test_abbreviated_header_joins_unique_full_name_and_takes_its_name():
    ms = [header("a1", "Rana K.", "rk@x.example"), llm("a2", 1, "Rana Karimi", "+1 Rana Karimi")]
    ids, name = entity_of(ms, "a1:h0")
    assert ids == {"a1:h0", "a2:m1"} and name == "Rana Karimi"


def test_jira_name_with_middle_initial_joins_header():
    ms = [header("a1", "Oskar Lind", "ol@x.example"), jira(1, "Oskar P. Lind")]
    assert entity_of(ms, "a1:h0")[0] == {"a1:h0", "jira:1:a0"}


def test_full_name_with_diminutive_joins():
    ms = [header("a1", "Robert Vance", "rv@x.example"), llm("a2", 1, "Bob Vance", "ask Bob Vance")]
    assert entity_of(ms, "a1:h0")[0] == {"a1:h0", "a2:m1"}


def test_sign_off_goes_to_sender():
    ms = [header("a1", "Nadia Brook", "nb@x.example"), llm("a1", 1, "Nadia", "Thanks, Nadia"),
          header("a2", "Nadia Stone", "ns@x.example")]
    assert entity_of(ms, "a1:m1")[0] == {"a1:h0", "a1:m1"}


def test_greeting_never_goes_to_sender_and_uses_unique_namesake():
    ms = [header("a1", "Chris Brook", "cb@x.example"), header("a2", "Christopher Stone", "cs@x.example"),
          llm("a2", 1, "Chris", "Hi Chris,")]
    assert entity_of(ms, "a2:m1")[0] == {"a1:h0", "a2:m1"}


def test_shared_first_name_without_local_evidence_stays_singleton():
    ms = [header("a1", "Dana Brook", "db@x.example"), header("a2", "Dana Stone", "ds@x.example"),
          header("a3", "Eli Moss", "em@x.example"), llm("a3", 1, "Dana", "Hi Dana,")]
    ids, name = entity_of(ms, "a3:m1")
    assert ids == {"a3:m1"} and name == "Dana"


def test_local_full_name_disambiguates_first_name():
    ms = [header("a1", "Dana Brook", "db@x.example"), header("a2", "Dana Stone", "ds@x.example"),
          header("a3", "Eli Moss", "em@x.example"), llm("a3", 1, "Dana", "Hi Dana,"),
          llm("a3", 2, "Dana Stone", "On Monday Dana Stone wrote:")]
    assert entity_of(ms, "a3:m1")[0] == {"a2:h0", "a3:m1", "a3:m2"}


def test_first_name_plus_initial_picks_matching_surname():
    ms = [header("a1", "Dana Brook", "db@x.example"), header("a2", "Dana Stone", "ds@x.example"),
          header("a3", "Eli Moss", "em@x.example"), llm("a3", 1, "Dana S", "and Dana S agreed")]
    assert entity_of(ms, "a3:m1")[0] == {"a2:h0", "a3:m1"}


def test_unique_first_name_across_corpus_is_assigned():
    ms = [header("a1", "Ines Okafor", "io@x.example"), header("a2", "Eli Moss", "em@x.example"),
          llm("a2", 1, "Ines", "Hi Ines,")]
    assert entity_of(ms, "a2:m1")[0] == {"a1:h0", "a2:m1"}


def test_address_in_context_links_mention():
    ms = [header("a1", "Ines Okafor", "io@x.example"), header("a2", "Ines Park", "ip@x.example"),
          llm("a3", 1, "Ines", "Ines <io@x.example> wrote:")]
    assert entity_of(ms, "a3:m1")[0] == {"a1:h0", "a3:m1"}


def test_username_matches_address_local_part():
    ms = [header("a1", "Ines Okafor", "iokafor9@x.example"), llm("a2", 1, "iokafor9", "@iokafor9 thanks")]
    assert entity_of(ms, "a2:m1")[0] == {"a1:h0", "a2:m1"}


def test_unknown_full_names_group_by_name():
    ms = [llm("a1", 1, "Tariq Noor", "Tariq Noor said"), llm("a2", 1, "tariq noor", "tariq noor said")]
    assert clusters(ms) == {frozenset({"a1:m1", "a2:m1"}): "Tariq Noor"}


def test_order_independent():
    ms = [header("a1", "Dana Brook", "db@x.example"), header("a2", "Dana B.", "db@x.example"),
          header("a3", "Dana Stone", "ds@x.example"), llm("a3", 1, "Dana", "Dana"),
          llm("a1", 1, "Dana", "Hi Dana"), llm("a4", 1, "Dana Brook", "Dana Brook")]
    shuffled = list(ms)
    random.Random(3).shuffle(shuffled)
    hints = hints_for(ms)
    a = {frozenset(m.mention_id for m in g): n for n, g in resolve_people(ms, hints)}
    b = {frozenset(m.mention_id for m in g): n for n, g in resolve_people(shuffled, hints)}
    assert a == b


def test_text_only_typo_joins_anchored_person():
    ms = [header("a1", "Bruna Cadenza", "bc@x.example"), llm("a2", 1, "Bruna Cadenzo", "thanks Bruna Cadenzo")]
    assert entity_of(ms, "a1:h0")[0] == {"a1:h0", "a2:m1"}


def test_typo_matching_two_people_does_not_join():
    ms = [header("a1", "Dana Brook", "db@x.example"), header("a2", "Dana Brooks", "ds@x.example"),
          llm("a3", 1, "Dana Brooke", "Dana Brooke said")]
    assert entity_of(ms, "a3:m1")[0] == {"a3:m1"}


def test_corpus_fallback_only_considers_list_participants():
    ms = [header("a1", "Ines Okafor", "io@x.example"), header("a2", "Eli Moss", "em@x.example"),
          llm("a2", 1, "Ines Park", "the paper by Ines Park"),
          header("a3", "Eli Moss", "em@x.example"), llm("a3", 1, "Ines", "Hi Ines,")]
    assert entity_of(ms, "a3:m1")[0] == {"a1:h0", "a3:m1"}


def test_corpus_fallback_uses_text_only_person_when_no_participant_fits():
    ms = [header("a2", "Eli Moss", "em@x.example"), llm("a2", 1, "Ines Park", "the paper by Ines Park"),
          header("a3", "Eli Moss", "em@x.example"), llm("a3", 1, "Ines", "as Ines wrote")]
    assert entity_of(ms, "a3:m1")[0] == {"a2:m1", "a3:m1"}
