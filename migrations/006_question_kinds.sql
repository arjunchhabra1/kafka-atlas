-- draft-v2: what kind of answer a draft has (date-centric drafts are capped per run), and the
-- new drafting outcomes (compound / too long / date cap / unviable / false null premise).
alter table questions add column answer_kind text
  check (answer_kind in ('date', 'entity', 'reason', 'yes_no', 'other'));

alter table evidence_attempts drop constraint evidence_attempts_outcome_check;
alter table evidence_attempts add constraint evidence_attempts_outcome_check check (outcome in (
  'kept', 'relabeled', 'shrunk', 'dropped_single_source', 'dropped_null_present',
  'dropped_shrunk', 'dropped_quota', 'failed', 'dropped_compound', 'dropped_too_long',
  'dropped_date_cap', 'dropped_unviable', 'dropped_null_premise'));
