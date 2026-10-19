from kgrag.ingest.clean import clean_body


def test_quoted_reply_removed():
    assert clean_body("I agree.\n> old text\n> more old\nBut see below.") == "I agree.\nBut see below."


def test_on_wrote_block_removed():
    text = "Thanks!\n\nOn Tue, Mar 5, 2024 at 10:00 AM Jun Rao <jun@x.com> wrote:\n> old"
    assert clean_body(text) == "Thanks!"


def test_on_wrote_wrapped_over_two_lines():
    text = "Thanks!\n\nOn Tue, Mar 5, 2024 at 10:00 AM Jun Rao\n<jun@x.com> wrote:\n> old\nmore"
    assert clean_body(text) == "Thanks!"


def test_signature_removed():
    assert clean_body("Body here.\n-- \nJun Rao\nConfluent") == "Body here."


def test_stack_trace_collapsed():
    text = (
        "It crashed:\n"
        "java.lang.NullPointerException: boom\n"
        "\tat org.apache.kafka.clients.consumer.KafkaConsumer.poll(KafkaConsumer.java:100)\n"
        "\tat org.apache.kafka.Foo.bar(Foo.java:5)\n"
        "Any ideas?"
    )
    out = clean_body(text)
    assert "[stack trace: NullPointerException in KafkaConsumer]" in out
    assert "KafkaConsumer.java" not in out
    assert out.startswith("It crashed:") and out.endswith("Any ideas?")


def test_prose_starting_with_at_is_kept():
    text = "at the moment we do not support this.\nat least not yet."
    assert clean_body(text) == text


def test_patch_collapsed():
    lines = ["diff --git a/x.java b/x.java", "index 1..2 100644", "--- a/x.java", "+++ b/x.java",
             "@@ -1,3 +1,4 @@", " ctx", "-old", "+new", "+new2", " ctx", " ctx", "+last"]
    assert len(lines) == 12
    out = clean_body("Patch below\n" + "\n".join(lines) + "\nDone")
    assert out == "Patch below\n[patch: 12 lines]\nDone"


def test_log_lines_collapsed():
    logs = [f"2024-03-05 10:00:{i:02d},000 INFO something happened {i}" for i in range(12)]
    assert clean_body("\n".join(logs)) == "[log: 12 lines]"


def test_short_log_run_kept():
    logs = [f"2024-03-05 10:00:{i:02d},000 INFO x" for i in range(9)]
    assert clean_body("\n".join(logs)) == "\n".join(logs)


def test_blank_lines_collapsed_and_trailing_stripped():
    assert clean_body("a\n\n\n\n\nb   \n\n") == "a\n\nb"
