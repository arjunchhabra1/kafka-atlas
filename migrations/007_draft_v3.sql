-- draft-v3: deterministic giveaway and Jira-date rejections, and the prompt version of every
-- drafting attempt (so cost estimates use attempts-per-kept measured under the current prompt).
alter table evidence_attempts add column prompt_version text;

alter table evidence_attempts drop constraint evidence_attempts_outcome_check;
alter table evidence_attempts add constraint evidence_attempts_outcome_check check (outcome in (
  'kept', 'relabeled', 'shrunk', 'dropped_single_source', 'dropped_null_present',
  'dropped_shrunk', 'dropped_quota', 'failed', 'dropped_compound', 'dropped_too_long',
  'dropped_date_cap', 'dropped_unviable', 'dropped_null_premise', 'dropped_giveaway',
  'dropped_jira_date'));
