create table questions (
  id text primary key,
  type text not null check (type in ('single_hop','inference','comparison','temporal','null')),
  question text not null,
  answer text not null,
  gold_doc_ids text[] not null default '{}',
  status text not null check (status in ('draft','kept','edited','dropped')),
  split text,
  draft_model text,
  prompt_version text,
  link text,
  origin text check (origin in ('sampled','relabeled')),
  key_terms text[],
  reviewed_at timestamptz
);

-- Every drafting attempt, so re-runs never pay twice for the same evidence set.
create table evidence_attempts (
  id serial primary key,
  type text not null,
  link text,
  evidence_doc_ids text[] not null,
  outcome text not null check (outcome in (
    'kept', 'relabeled', 'shrunk', 'dropped_single_source', 'dropped_null_present',
    'dropped_shrunk', 'dropped_quota', 'failed')),
  question_id text,
  attempted_at timestamptz default now(),
  unique (type, evidence_doc_ids, link)
);
